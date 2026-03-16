# LLM Sandbox → OpenAI API Proxy

A lightweight proxy that exposes an **OpenAI-compatible API** (`/v1/chat/completions`) backed by the UCSB LLM Sandbox Bot API. Point any tool that speaks OpenAI format at this proxy and it translates under the hood.

## Why

The LLM Sandbox Bot API doesn't accept the standard `messages[]` array format that most tools expect. It takes a single text message per request, and responses require polling rather than returning directly. This proxy bridges that gap — it accepts OpenAI-format requests, flattens the messages into a single prompt, handles the polling, and returns a standard OpenAI-format response.

## Quick Start

```bash
# Clone and install
git clone https://github.com/bhill00/llmsandbox-openai-proxy.git
cd llmsandbox-openai-proxy
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run (set your credentials)
BEDROCK_API_URL=https://your-api-url/api \
BEDROCK_API_KEY=your-key \
uvicorn server:app --port 8780
```

The proxy is now running at `http://127.0.0.1:8780/v1`.

## Usage

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8780/v1",
    api_key="not-needed",  # auth is handled by the proxy
)

response = client.chat.completions.create(
    model="claude-v4.5-sonnet",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain Python decorators in 3 sentences."},
    ],
)

print(response.choices[0].message.content)
```

### curl

```bash
curl http://127.0.0.1:8780/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-v4.5-sonnet",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://127.0.0.1:8780/v1",
    api_key="not-needed",
    model="claude-v4.5-sonnet",
)

response = llm.invoke("What is NIST 800-171?")
print(response.content)
```

### Vision (image input)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8780/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="claude-v4.5-sonnet",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/photo.png"}},
            {"type": "text", "text": "What's in this image?"},
        ],
    }],
)
print(response.choices[0].message.content)
```

Base64 data URIs also work: `"url": "data:image/png;base64,iVBOR..."`.

### Aider

```bash
aider --openai-api-base http://127.0.0.1:8780/v1 --openai-api-key not-needed --model claude-v4.5-sonnet
```

## Configuration

Environment variables:

- `BEDROCK_API_URL` (required) — LLM Sandbox Bot API URL
- `BEDROCK_API_KEY` (required) — LLM Sandbox API key
- `POLL_INTERVAL` (default: 2) — seconds between polling attempts
- `POLL_TIMEOUT` (default: 60) — max seconds to wait for a response
- `DEFAULT_MODEL` (default: claude-v4.5-sonnet) — model used when none specified

## Model Names

The Sandbox models use names like `claude-v4.5-sonnet`. Use these directly:

- `claude-v4.5-sonnet` — Claude Sonnet 4.5
- `claude-v4-sonnet` — Claude Sonnet 4
- `claude-v3.5-sonnet` — Claude Sonnet 3.5

Check with your sandbox administrator for the full list of available models (including Amazon Nova and others). Any model name is passed through to the Sandbox as-is.

**Convenience aliases:** If you have existing code that uses OpenAI model names, the proxy maps them automatically so you don't have to change your code:

- `gpt-4`, `gpt-4o`, `gpt-4-turbo` → routes to `claude-v4.5-sonnet`
- `gpt-3.5-turbo` → routes to `claude-v3.5-sonnet`

These are just aliases for convenience — no GPT models are available through the Sandbox.

## Compatibility

### What works

- `/v1/chat/completions` — full messages[] array with system/user/assistant roles
- `/v1/models` — list available models
- Multi-turn conversations via messages array
- **Vision / image inputs** — supports both base64 data URIs and image URLs in OpenAI's `image_url` format. The proxy translates them to the Sandbox's native image content type.
- Streaming (`stream: true`) — faked by returning the complete response as SSE chunks. Tools won't break, but you don't get real token-by-token output.
- System prompts, model selection, basic parameters

### What doesn't work

- **Structured function calling** — the proxy does not return OpenAI-style `tool_calls` in responses. If your code checks `response.choices[0].message.tool_calls`, it won't find anything. See the note on tool use below.
- **Embeddings** (`/v1/embeddings`) — completely different API, not available through the Sandbox
- **Files / Assistants / Threads API** — OpenAI-specific features with no Sandbox equivalent
- **Accurate token usage** — `usage.prompt_tokens` is always 0 (the Sandbox doesn't report input counts). `completion_tokens` is a rough estimate (~4 chars per token). Do not rely on these for cost tracking.
- **Logprobs, batching** — not exposed by the Sandbox

### A note on tool use / function calling

There are three levels of tool use to understand:

**Server-side tools (works transparently):** If your bot has Agent mode enabled with tools like Internet Search or Knowledge Base, the backend executes them automatically. The client sends a normal message, the backend decides whether to search the web or query documents, and returns the final answer. No special client code needed — tested and confirmed working through the proxy.

**Prompt-engineered client-side tools (works with caveats):** The model will respond with structured JSON when asked to use tools via the prompt. You define tools in your system prompt, the model responds with JSON indicating which tool to call, your code parses it and executes locally, then sends the result back. Less reliable than native structured `tool_calls` (the model might occasionally break format), but functional for most use cases.

**OpenAI structured tool calling protocol (does not work):** Sending a `tools` array in the request and getting back `tool_calls` objects in the response with guaranteed-valid JSON. The Bot API doesn't expose a field for client-provided tool definitions. Tools that depend on this protocol (Cursor's edit mode, some LangChain agents) will not work.

Most projects using the OpenAI API don't use function calling at all. Chat completions, RAG pipelines, code generation, content workflows, and data processing scripts all work fine without it.

### Tool compatibility at a glance

- **OpenAI Python SDK** — works well for chat completions and vision
- **LangChain / LlamaIndex** — chains work. Agents that need structured function calling don't.
- **Aider** — works well (uses chat completions, parses code from text responses)
- **Open WebUI** — basic chat works. Plugin/tool features that rely on function calling won't.
- **Continue (VS Code)** — basic chat works. Autocomplete and codebase features need function calling/embeddings.
- **Cursor** — core edit/compose features rely on structured function calling. Basic chat tab might work.
- **Custom scripts / notebooks** — best use case. Change `base_url` and go.

## Important: Token Cost

The Sandbox has **no prompt caching**. Every token is full price, every turn. The proxy flattens your entire messages[] array into a single prompt, so longer conversation histories = proportionally more tokens per request. Unlike the standard OpenAI API where prompt caching discounts repeated prefixes, here every token in every request costs the same.

The client (Aider, LangChain, your script) manages the messages array and sends it in full each time. The proxy does not do any context compression — it's a format translator only. If your client sends 50 messages in the array, all 50 get flattened and sent.

Keep conversations short. Reset often. See the [llmsandbox-extension README](https://github.com/bhill00/llmsandbox-extension#understanding-context-tokens-and-cost) for a detailed cost analysis.

## License

MIT
