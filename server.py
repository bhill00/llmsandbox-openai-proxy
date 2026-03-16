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
from typing import Dict, List, Optional, Tuple, Union

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

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

if not API_URL or not API_KEY:
    raise RuntimeError("BEDROCK_API_URL and BEDROCK_API_KEY must be set")

BOT_HEADERS = {"x-api-key": API_KEY, "Content-Type": "application/json"}

# ---------------------------------------------------------------------------
# Model mapping: OpenAI model names → Sandbox model names
# Users can use either the OpenAI-style name or the Sandbox name directly.
# ---------------------------------------------------------------------------
MODEL_MAP = {
    # Claude models
    "gpt-4": DEFAULT_MODEL,
    "gpt-4o": DEFAULT_MODEL,
    "gpt-4-turbo": DEFAULT_MODEL,
    "gpt-3.5-turbo": "claude-v3.5-sonnet",
    "claude-sonnet-4-5": "claude-v4.5-sonnet",
    "claude-sonnet-4": "claude-v4-sonnet",
    "claude-3.5-sonnet": "claude-v3.5-sonnet",
}


def resolve_model(model: str) -> str:
    """Map an OpenAI-style model name to a Sandbox model name."""
    return MODEL_MAP.get(model, model)


# ---------------------------------------------------------------------------
# Session state: one conversation per proxy instance (simple approach)
# For multi-user support, you'd key by a session/API-key header.
# ---------------------------------------------------------------------------
conversations: Dict[str, Dict] = {}
# conversations[conv_id] = {"parent_message_id": str|None}


def get_or_create_conversation(conv_id: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """Return (conversation_id, parent_message_id)."""
    if conv_id and conv_id in conversations:
        return conv_id, conversations[conv_id]["parent_message_id"]
    new_id = conv_id or str(uuid.uuid4())
    conversations[new_id] = {"parent_message_id": None}
    return new_id, None


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


def poll_for_reply(conversation_id: str, user_message_id: str) -> str:
    """Poll GET endpoint until the assistant reply appears."""
    for attempt in range(POLL_MAX_ATTEMPTS):
        time.sleep(POLL_INTERVAL)
        resp = requests.get(
            f"{API_URL}/conversation/{conversation_id}",
            headers=BOT_HEADERS,
        )
        resp.raise_for_status()
        data = resp.json()

        msg_map = data.get("messageMap", {})
        user_msg = msg_map.get(user_message_id, {})
        children = user_msg.get("children", [])
        if children:
            assistant_msg = msg_map.get(children[0], {})
            if assistant_msg.get("role") == "assistant":
                return assistant_msg.get("content", [{}])[0].get("body", "")

        # Fallback: check lastMessageId
        last_id = data.get("lastMessageId")
        last_msg = msg_map.get(last_id, {})
        if last_msg.get("role") == "assistant":
            return last_msg.get("content", [{}])[0].get("body", "")

        log.debug("Poll attempt %d/%d — no reply yet", attempt + 1, POLL_MAX_ATTEMPTS)

    raise HTTPException(status_code=504, detail="Timed out waiting for response from LLM Sandbox")


def call_sandbox(
    messages: List[Message],
    model: str,
    conversation_id: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Send a request to the Sandbox API and return (reply, model_used, conversation_id)."""
    sandbox_model = resolve_model(model)
    conv_id, parent_message_id = get_or_create_conversation(conversation_id)

    content_blocks = assemble_content(messages)
    message_id = str(uuid.uuid4())

    # Estimate tokens from text blocks only
    text_total = sum(
        len(b.get("body", "")) for b in content_blocks if b.get("contentType") == "text"
    )
    image_count = sum(1 for b in content_blocks if b.get("contentType") == "image")

    log.info(
        "Sending to Sandbox: model=%s, conv=%s, ~%d tokens, %d images",
        sandbox_model, conv_id, text_total // CHARS_PER_TOKEN, image_count,
    )

    payload = {
        "conversation_id": conv_id,
        "message": {
            "role": "user",
            "content": content_blocks,
            "model": sandbox_model,
            "parent_message_id": parent_message_id,
            "message_id": message_id,
        },
        "continue_generate": False,
        "enable_reasoning": False,
    }

    post_resp = requests.post(
        f"{API_URL}/conversation", headers=BOT_HEADERS, json=payload
    )
    post_resp.raise_for_status()
    server_message_id = post_resp.json().get("messageId", message_id)

    reply = poll_for_reply(conv_id, server_message_id)

    # Update conversation state for threading
    conversations[conv_id]["parent_message_id"] = server_message_id

    return reply, sandbox_model, conv_id


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
        reply, model_used, conv_id = call_sandbox(
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
        "claude-v4.5-sonnet",
        "claude-v4-sonnet",
        "claude-v3.5-sonnet",
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
