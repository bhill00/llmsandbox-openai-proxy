# LLM Sandbox → OpenAI API Proxy

A lightweight proxy that exposes an **OpenAI-compatible API** (`/v1/chat/completions`) backed by the UCSB LLM Sandbox Bot API. Point any tool that speaks OpenAI format at this proxy and it translates under the hood.

## Why

The LLM Sandbox Bot API doesn't accept the standard `messages[]` array format that most tools expect. It takes a single message per request, and responses require polling rather than returning directly. This proxy bridges that gap — it accepts OpenAI-format requests, translates them to the Sandbox format, handles the polling, and returns a standard OpenAI-format response.

The proxy supports two memory modes:

- **Server mode** (default): Sends only the latest user message each turn. The server maintains conversation history automatically. This is token-efficient — turn N costs the same as turn 1.
- **Client mode**: Flattens the entire `messages[]` array into a single prompt each turn, with no server-side memory. Use this if server-side conversation handling is causing problems.

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
- `MEMORY_MODE` (default: `server`) — `server` sends only the last user message and lets the server track history; `client` flattens the full messages[] array into every request. Use `client` for tools like Cline or Aider that manage their own conversation history — server mode would double the context. Use `server` for simple scripts or one-shot API calls.
- `POLL_TIMEOUT` (default: 60) — max seconds to wait for a response
- `POLL_INITIAL_INTERVAL` (default: 0.3) — initial polling delay in seconds (adaptive backoff starts here)
- `POLL_BACKOFF_MULTIPLIER` (default: 1.5) — multiplier for exponential backoff between polls
- `POLL_MAX_INTERVAL` (default: 5.0) — maximum polling interval in seconds
- `DEFAULT_MODEL` (default: claude-v4.5-sonnet) — model used when none specified

## Model Names

Use the Sandbox model names directly:

| Sandbox name | Model |
|---|---|
| `claude-v4.6-opus` | Claude 4.6 (Opus) |
| `claude-v4.5-opus` | Claude 4.5 Opus |
| `claude-v4.5-sonnet` | Claude 4.5 Sonnet |
| `claude-v4.5-haiku` | Claude 4.5 Haiku |
| `amazon-nova-pro` | Amazon Nova Pro |
| `amazon-nova-lite` | Amazon Nova Lite |
| `amazon-nova-micro` | Amazon Nova Micro |
| `qwen3-32b` | Qwen3 32B |

Any model name is passed through to the Sandbox as-is, so new models work without a proxy update.

**Convenience aliases:** If you have existing code that uses OpenAI or Anthropic API model names, the proxy maps them automatically:

- `gpt-4`, `gpt-4o`, `gpt-4-turbo` → `claude-v4.5-sonnet`
- `gpt-3.5-turbo` → `claude-v4.5-haiku`
- `claude-opus-4-6` → `claude-v4.6-opus`
- `claude-opus-4-5` → `claude-v4.5-opus`
- `claude-sonnet-4-5` → `claude-v4.5-sonnet`
- `claude-haiku-4-5` → `claude-v4.5-haiku`

These are just aliases for convenience — no GPT models are available through the Sandbox.

## Compatibility

### What works

- `/v1/chat/completions` — full messages[] array with system/user/assistant roles
- `/v1/models` — list available models
- Multi-turn conversations via messages array
- **Vision / image inputs** — supports both base64 data URIs and image URLs in OpenAI's standard `image_url` format. The proxy translates them to the Sandbox's native image content type.
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
- **Cline (VS Code)** — experimental. Requires `MEMORY_MODE=client`. Basic tasks work but polling latency can cause timeouts on longer exchanges.
- **Aider** — works well (uses chat completions, parses code from text responses). Use `MEMORY_MODE=client`.
- **Open WebUI** — basic chat works. Plugin/tool features that rely on function calling won't.
- **Continue (VS Code)** — basic chat works. Autocomplete and codebase features need function calling/embeddings.
- **Cursor** — core edit/compose features rely on structured function calling. Basic chat tab might work.
- **Custom scripts / notebooks** — best use case. Change `base_url` and go.

## Important: Bot Configuration

The Sandbox does **not** support per-request inference parameters. Values like `temperature`, `max_tokens`, and `top_p` sent by clients (LangChain, Cline, etc.) are accepted by the proxy but **ignored by the Sandbox** — the bot's configured settings always win. You must configure these in the Sandbox bot dashboard.

The proxy sends `enableReasoning: false` in each request to suppress extended thinking, but the bot's reasoning budget setting may override this.

### Recommended bot settings by use case

| Setting | RAG / Analysis | Coding (Cline/Aider) | General Chat |
|---|---|---|---|
| Max generation | 16000 | 16000 | 4096 |
| Temperature | 0.1–0.2 | 0.2 | 0.6 |
| Top-p | 0.95 | 0.95 | 0.999 |
| Top-k | 128 | 128 | 128 |
| Reasoning budget | 1024 (minimum) | 1024 (minimum) | 1024+ |

**For RAG pipelines and coding assistants**, low temperature (0.1–0.2) is critical — higher values cause the model to hallucinate facts and deviate from tool-use protocols. If you're using one bot for multiple purposes, 0.2 is a reasonable compromise.

**If per-request parameter control is needed**, the upstream [bedrock-chat](https://github.com/aws-samples/bedrock-chat) project would need to be modified to accept `generation_params` on the `/conversation` endpoint, not just on bot creation/modification. Consider requesting this from your Sandbox admin.

## Important: Token Cost

The Sandbox has **no prompt caching**. Every token is full price, every turn.

**Server mode** (default) mitigates this: only the latest user message is sent each turn, and the server reconstructs history from its own storage. Turn 10 costs roughly the same as turn 1. This is the recommended mode for most use cases.

**Client mode** flattens your entire `messages[]` array into a single prompt every turn, so costs grow linearly with conversation length. Use this if server-side conversation handling is causing problems.

In either mode, keep conversations focused. See the [llmsandbox-extension README](https://github.com/bhill00/llmsandbox-extension#understanding-context-tokens-and-cost) for a detailed cost analysis.

## License

MIT
