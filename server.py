"""
LLM Sandbox → OpenAI API Proxy

Exposes an OpenAI-compatible /v1/chat/completions endpoint that translates
requests to the UCSB LLM Sandbox Bot API format.

Usage:
    BEDROCK_API_URL=https://... BEDROCK_API_KEY=... uvicorn server:app --port 8780
"""

from __future__ import annotations

import base64
import os
import uuid
import time
import logging
from typing import List, Optional, Tuple, Union

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("llmsandbox-proxy")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_URL = os.environ.get("BEDROCK_API_URL")
API_KEY = os.environ.get("BEDROCK_API_KEY")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "2"))
POLL_TIMEOUT = int(os.environ.get("POLL_TIMEOUT", "60"))
POLL_MAX_ATTEMPTS = max(1, POLL_TIMEOUT // max(1, POLL_INTERVAL))
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "claude-v4.5-sonnet")
MEMORY_MODE = os.environ.get("MEMORY_MODE", "server")  # "server" or "client"

if not API_URL or not API_KEY:
    raise RuntimeError("BEDROCK_API_URL and BEDROCK_API_KEY must be set")

BOT_HEADERS = {"x-api-key": API_KEY, "Content-Type": "application/json"}

# ---------------------------------------------------------------------------
# Model mapping: OpenAI model names → Sandbox model names
# Users can use either the OpenAI-style name or the Sandbox name directly.
# ---------------------------------------------------------------------------
MODEL_MAP = {
    # OpenAI-style names → Sandbox defaults
    "gpt-4": DEFAULT_MODEL,
    "gpt-4o": DEFAULT_MODEL,
    "gpt-4-turbo": DEFAULT_MODEL,
    "gpt-3.5-turbo": "claude-v4.5-haiku",
    # Anthropic API names → Sandbox names
    "claude-opus-4-6": "claude-v4.6-opus",
    "claude-opus-4-5": "claude-v4.5-opus",
    "claude-sonnet-4-5": "claude-v4.5-sonnet",
    "claude-haiku-4-5": "claude-v4.5-haiku",
}


def resolve_model(model: str) -> str:
    """Map an OpenAI-style model name to a Sandbox model name."""
    return MODEL_MAP.get(model, model)


# ---------------------------------------------------------------------------
# Session state
# Server mode: track conversationId so multi-turn reuses the same thread.
# Client mode: no state — every call is a fresh conversation.
# ---------------------------------------------------------------------------
server_conversation_id: Optional[str] = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="LLM Sandbox OpenAI Proxy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# OpenAI-compatible request/response models
# ---------------------------------------------------------------------------
class Message(BaseModel):
    role: str
    content: Union[str, list, None] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: List[Message]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    # We accept but ignore these OpenAI-specific fields
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    n: Optional[int] = None
    user: Optional[str] = None


# ---------------------------------------------------------------------------
# Core: assemble messages and call the Sandbox API
# ---------------------------------------------------------------------------
CHARS_PER_TOKEN = 4  # rough estimate for token counting


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def _fetch_image_as_base64(url: str) -> Tuple[str, str]:
    """Fetch an image URL and return (base64_data, media_type)."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "image/png").split(";")[0].strip()
    b64 = base64.b64encode(resp.content).decode("utf-8")
    return b64, content_type


def _parse_data_uri(uri: str) -> Tuple[str, str]:
    """Parse a data:image/...;base64,... URI into (base64_data, media_type)."""
    # data:image/png;base64,iVBOR...
    header, b64_data = uri.split(",", 1)
    media_type = header.split(":")[1].split(";")[0]
    return b64_data, media_type


def _extract_image_content(block: dict) -> Optional[dict]:
    """Convert an OpenAI image_url content block to Sandbox image format."""
    image_url = block.get("image_url", {})
    url = image_url.get("url", "")

    if not url:
        return None

    try:
        if url.startswith("data:"):
            b64_data, media_type = _parse_data_uri(url)
        else:
            b64_data, media_type = _fetch_image_as_base64(url)

        return {
            "contentType": "image",
            "mediaType": media_type,
            "body": b64_data,
        }
    except Exception as exc:
        log.warning("Failed to process image: %s", exc)
        return None


def extract_last_user_message(messages: List[Message]) -> List[dict]:
    """Extract only the last user message for server memory mode.

    The server reconstructs full history via trace_to_root(), so we only
    need to send the newest user turn.  No role prefix — the server knows
    it comes from the user.
    """
    for msg in reversed(messages):
        if msg.role != "user":
            continue
        raw = msg.content
        if raw is None:
            continue

        content_blocks: List[dict] = []
        if isinstance(raw, str):
            content_blocks.append({"contentType": "text", "body": raw})
        elif isinstance(raw, list):
            text_parts: List[str] = []
            for block in raw:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict):
                    block_type = block.get("type", "")
                    if block_type == "text":
                        text_parts.append(block.get("text", ""))
                    elif block_type == "image_url":
                        if text_parts:
                            content_blocks.append({
                                "contentType": "text",
                                "body": " ".join(text_parts),
                            })
                            text_parts = []
                        img = _extract_image_content(block)
                        if img:
                            content_blocks.append(img)
            if text_parts:
                content_blocks.append({
                    "contentType": "text",
                    "body": " ".join(text_parts),
                })
        return content_blocks

    return [{"contentType": "text", "body": ""}]


def assemble_content(messages: List[Message]) -> List[dict]:
    """Convert an OpenAI messages[] array into a Sandbox content array.

    This is the key translation layer. The Sandbox API expects a single
    message with a content array, so we flatten the conversation into
    text blocks with role prefixes, preserving images inline.
    """
    content_blocks = []

    for msg in messages:
        raw = msg.content

        if raw is None:
            continue

        # Determine role prefix
        if msg.role == "system":
            prefix = "System instructions"
        elif msg.role == "user":
            prefix = "User"
        elif msg.role == "assistant":
            prefix = "Assistant"
        else:
            prefix = msg.role

        if isinstance(raw, str):
            # Simple text content
            content_blocks.append({"contentType": "text", "body": f"{prefix}: {raw}"})

        elif isinstance(raw, list):
            # Content array (multimodal) — may contain text and images
            text_parts = []
            for block in raw:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict):
                    block_type = block.get("type", "")

                    if block_type == "text":
                        text_parts.append(block.get("text", ""))

                    elif block_type == "image_url":
                        # Flush accumulated text first
                        if text_parts:
                            content_blocks.append({
                                "contentType": "text",
                                "body": f"{prefix}: {' '.join(text_parts)}",
                            })
                            text_parts = []

                        img = _extract_image_content(block)
                        if img:
                            content_blocks.append(img)

            # Flush remaining text
            if text_parts:
                content_blocks.append({
                    "contentType": "text",
                    "body": f"{prefix}: {' '.join(text_parts)}",
                })

    return content_blocks


def _is_turn_complete(msg: dict) -> bool:
    """Check if an assistant message represents a complete turn.

    During agent tool-use loops, the model may produce intermediate messages
    (acknowledgments, tool_use blocks) before the final answer. We need to
    wait for the turn to actually be done.
    """
    # Check for explicit stop_reason / stopReason field
    stop_reason = msg.get("stop_reason") or msg.get("stopReason") or ""
    if stop_reason == "tool_use":
        return False  # Model wants to call a tool — not done yet
    if stop_reason in ("end_turn", "stop", "max_tokens"):
        return True  # Explicitly complete

    # Check content — if only tool_use blocks and no text, likely intermediate
    content = msg.get("content", [])
    has_text = any(
        c.get("contentType") == "text" and c.get("body", "").strip()
        for c in content
    )
    has_tool_use = any(c.get("contentType") == "toolUse" for c in content)

    if has_tool_use and not has_text:
        return False  # Only tool calls, no text — intermediate

    # If there's text content and no stop_reason field, assume complete
    return has_text


def poll_for_reply(conversation_id: str, message_id: str) -> str:
    """Poll the per-message endpoint until a complete assistant reply appears."""
    for attempt in range(POLL_MAX_ATTEMPTS):
        time.sleep(POLL_INTERVAL)
        resp = requests.get(
            f"{API_URL}/conversation/{conversation_id}/{message_id}",
            headers=BOT_HEADERS,
        )
        # 404 means the reply message doesn't exist yet — keep polling
        if resp.status_code == 404:
            log.debug("Poll attempt %d/%d — reply not created yet", attempt + 1, POLL_MAX_ATTEMPTS)
            continue
        resp.raise_for_status()
        data = resp.json()

        msg = data.get("message", {})
        if msg.get("role") == "assistant":
            if not _is_turn_complete(msg):
                log.debug("Poll attempt %d/%d — assistant message not complete (tool use in progress)", attempt + 1, POLL_MAX_ATTEMPTS)
                continue

            # Extract text content from the complete response
            content = msg.get("content", [])
            text_parts = [
                c.get("body", "")
                for c in content
                if c.get("contentType") == "text" and c.get("body", "").strip()
            ]
            if text_parts:
                return "\n\n".join(text_parts)

        log.debug("Poll attempt %d/%d — no reply yet", attempt + 1, POLL_MAX_ATTEMPTS)

    raise HTTPException(status_code=504, detail="Timed out waiting for response from LLM Sandbox")


def call_sandbox(
    messages: List[Message],
    model: str,
) -> Tuple[str, str]:
    """Send a request to the Sandbox API and return (reply, model_used)."""
    global server_conversation_id

    sandbox_model = resolve_model(model)

    if MEMORY_MODE == "server":
        content_blocks = extract_last_user_message(messages)
        conv_id = server_conversation_id or str(uuid.uuid4())
    else:
        content_blocks = assemble_content(messages)
        conv_id = str(uuid.uuid4())  # fresh conversation every call

    # Estimate tokens from text blocks only
    text_total = sum(
        len(b.get("body", "")) for b in content_blocks if b.get("contentType") == "text"
    )
    image_count = sum(1 for b in content_blocks if b.get("contentType") == "image")

    log.info(
        "Sending to Sandbox [%s mode]: model=%s, conv=%s, ~%d tokens, %d images",
        MEMORY_MODE, sandbox_model, conv_id, text_total // CHARS_PER_TOKEN, image_count,
    )

    payload = {
        "message": {
            "role": "user",
            "parent_message_id": None,
            "content": content_blocks,
            "model": sandbox_model,
        },
    }

    post_resp = requests.post(
        f"{API_URL}/conversation", headers=BOT_HEADERS, json=payload
    )
    post_resp.raise_for_status()
    resp_data = post_resp.json()
    server_message_id = resp_data.get("messageId")
    conv_id = resp_data.get("conversationId", conv_id)

    reply = poll_for_reply(conv_id, server_message_id)

    if MEMORY_MODE == "server":
        server_conversation_id = conv_id

    return reply, sandbox_model


# ---------------------------------------------------------------------------
# Build OpenAI-format response
# ---------------------------------------------------------------------------
def build_completion_response(
    reply: str,
    model: str,
    request_id: str,
) -> dict:
    """Build an OpenAI-compatible chat completion response."""
    prompt_tokens = 0  # We can't know exact token counts
    completion_tokens = estimate_tokens(reply)

    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": reply,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def build_streaming_response(reply: str, model: str, request_id: str):
    """Yield SSE chunks that mimic OpenAI streaming format.

    Since the Sandbox API doesn't stream, we fake it by sending the
    complete response as a single content chunk.
    """
    import json

    chunk_id = f"chatcmpl-{request_id}"

    # Role chunk
    yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

    # Content chunk — send the full reply at once
    yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {'content': reply}, 'finish_reason': None}]})}\n\n"

    # Stop chunk
    yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    request_id = str(uuid.uuid4())[:8]

    try:
        reply, model_used = call_sandbox(
            messages=req.messages,
            model=req.model,
        )
    except requests.exceptions.HTTPError as e:
        raise HTTPException(
            status_code=e.response.status_code if e.response else 502,
            detail=f"Sandbox API error: {e}",
        )

    if req.stream:
        return StreamingResponse(
            build_streaming_response(reply, model_used, request_id),
            media_type="text/event-stream",
        )

    return build_completion_response(reply, model_used, request_id)


@app.get("/v1/models")
def list_models():
    """Return available models in OpenAI format."""
    models = [
        "claude-v4.6-opus",
        "claude-v4.5-opus",
        "claude-v4.5-sonnet",
        "claude-v4.5-haiku",
        "amazon-nova-pro",
        "amazon-nova-lite",
        "amazon-nova-micro",
        "qwen3-32b",
    ]
    return {
        "object": "list",
        "data": [
            {
                "id": m,
                "object": "model",
                "created": 0,
                "owned_by": "ucsb-llmsandbox",
            }
            for m in models
        ],
    }


@app.get("/v1/models/{model_id}")
def get_model(model_id: str):
    """Return model info in OpenAI format."""
    return {
        "id": model_id,
        "object": "model",
        "created": 0,
        "owned_by": "ucsb-llmsandbox",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "message": "LLM Sandbox OpenAI Proxy",
        "docs": "/docs",
        "openai_base_url": "http://127.0.0.1:8780/v1",
    }
