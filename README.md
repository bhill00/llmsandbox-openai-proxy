# LLM Sandbox → OpenAI API Proxy

A lightweight proxy that exposes an **OpenAI-compatible API** (`/v1/chat/completions`) backed by the UCSB LLM Sandbox Bot API. Point any tool that speaks OpenAI format at this proxy and it translates under the hood.

## Why

The LLM Sandbox Bot API is stateless and async — it doesn't accept the standard `messages[]` array format that most tools expect. This proxy bridges that gap so you can use the Sandbox from tools, libraries, and scripts that were built for the OpenAI API.

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

You can use either Sandbox model names or common OpenAI names:

- `claude-v4.5-sonnet` / `claude-sonnet-4-5` / `gpt-4` / `gpt-4o` / `gpt-4-turbo` → Claude Sonnet 4.5
- `claude-v4-sonnet` / `claude-sonnet-4` → Claude Sonnet 4
- `claude-v3.5-sonnet` / `claude-3.5-sonnet` / `gpt-3.5-turbo` → Claude Sonnet 3.5

Or pass any model name the Sandbox supports directly — it will be forwarded as-is if there's no mapping. Check with your sandbox administrator for the full list of available models.

## Compatibility

### What works

- `/v1/chat/completions` — full messages[] array with system/user/assistant roles
- `/v1/models` — list available models
- Multi-turn conversations via messages array
- Streaming (`stream: true`) — faked by returning the complete response as SSE chunks. Tools won't break, but you don't get real token-by-token output.
- System prompts, model selection, basic parameters

### What doesn't work

- **Function calling / tool use** — the Sandbox API has no native tool schema support. Tools that rely on structured function call responses (Cursor's edit mode, some LangChain agents) will not work.
- **Vision / image inputs** — multimodal content is not supported through the proxy
- **Embeddings** (`/v1/embeddings`) — completely different API, not available through the Sandbox
- **Files / Assistants / Threads API** — OpenAI-specific features with no Sandbox equivalent
- **Accurate token usage** — `usage.prompt_tokens` is always 0 (the Sandbox doesn't report input counts). `completion_tokens` is a rough estimate (~4 chars per token). Do not rely on these for cost tracking.
- **Logprobs, batching** — not exposed by the Sandbox

### Tool compatibility at a glance

- **OpenAI Python SDK** — works well for chat completions
- **LangChain / LlamaIndex** — works for basic chains. Breaks if agents need function calling.
- **Aider** — works reasonably well (uses chat completions, parses code from text)
- **Open WebUI** — basic chat works. Plugin/tool features won't.
- **Continue (VS Code)** — basic chat works. Autocomplete and codebase features need function calling/embeddings.
- **Cursor** — core edit/compose features rely heavily on streaming and function calling. Basic chat tab might work.
- **Custom scripts / notebooks** — best use case. Change `base_url` and go.

## Important: Token Cost

The Sandbox has **no prompt caching**. Every token is full price, every turn. The proxy flattens your entire messages[] array into a single prompt, so longer conversation histories = proportionally more tokens per request. Unlike the standard OpenAI API where prompt caching discounts repeated prefixes, here every token in every request costs the same.

The client (Aider, LangChain, your script) manages the messages array and sends it in full each time. The proxy does not do any context compression — it's a format translator only. If your client sends 50 messages in the array, all 50 get flattened and sent.

Keep conversations short. Reset often. See the [llmsandbox-extension README](https://github.com/bhill00/llmsandbox-extension#understanding-context-tokens-and-cost) for a detailed cost analysis.

## License

MIT
