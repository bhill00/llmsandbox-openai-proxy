"""
Quick smoke tests for the OpenAI-compatible proxy.
Requires the proxy to be running on http://127.0.0.1:8780

Usage:
    python test_proxy.py
"""

import time
import json
import requests

BASE = "http://127.0.0.1:8780"

def test(name, fn):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    try:
        start = time.time()
        fn()
        elapsed = time.time() - start
        print(f"  PASS ({elapsed:.1f}s)")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_health():
    r = requests.get(f"{BASE}/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    print("  Health OK")


def test_list_models():
    r = requests.get(f"{BASE}/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    models = [m["id"] for m in data["data"]]
    print(f"  Models: {models}")
    assert len(models) > 0
    assert "claude-v4.5-sonnet" in models


def test_get_model():
    r = requests.get(f"{BASE}/v1/models/claude-v4.5-sonnet")
    assert r.status_code == 200
    data = r.json()
    assert data["id"] == "claude-v4.5-sonnet"
    assert data["object"] == "model"
    print(f"  Model info: {data}")


def test_chat_basic():
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": "claude-v4.5-sonnet",
        "messages": [{"role": "user", "content": "Reply with just the word 'pong'"}],
    })
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    reply = data["choices"][0]["message"]["content"]
    print(f"  Reply: {reply[:100]}")
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert "usage" in data


def test_chat_system_prompt():
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": "claude-v4.5-sonnet",
        "messages": [
            {"role": "system", "content": "You are a pirate. Always say 'Arrr' in your response."},
            {"role": "user", "content": "Hello"},
        ],
    })
    assert r.status_code == 200
    reply = r.json()["choices"][0]["message"]["content"]
    print(f"  Reply: {reply[:200]}")


def test_chat_streaming():
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": "claude-v4.5-sonnet",
        "messages": [{"role": "user", "content": "Reply with just the word 'stream'"}],
        "stream": True,
    }, stream=True)
    assert r.status_code == 200
    assert "text/event-stream" in r.headers.get("content-type", "")

    chunks = []
    content = ""
    for line in r.iter_lines(decode_unicode=True):
        if line.startswith("data: "):
            payload = line[6:]
            if payload == "[DONE]":
                chunks.append("[DONE]")
                break
            chunk = json.loads(payload)
            assert chunk["object"] == "chat.completion.chunk"
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                content += delta["content"]
            chunks.append(chunk)

    print(f"  Chunks received: {len(chunks)}")
    print(f"  Content: {content[:100]}")
    assert len(chunks) >= 3  # role + content + stop + DONE


def test_chat_model_alias():
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Reply with just the word 'alias'"}],
    })
    assert r.status_code == 200
    data = r.json()
    # gpt-4 should map to claude-v4.5-sonnet
    print(f"  Model used: {data['model']}")
    print(f"  Reply: {data['choices'][0]['message']['content'][:100]}")


def test_response_structure():
    """Verify all required OpenAI response fields are present."""
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": "claude-v4.5-sonnet",
        "messages": [{"role": "user", "content": "Say hi"}],
    })
    data = r.json()

    # Top level
    assert "id" in data and data["id"].startswith("chatcmpl-")
    assert data["object"] == "chat.completion"
    assert "created" in data
    assert "model" in data

    # Choices
    choice = data["choices"][0]
    assert choice["index"] == 0
    assert "message" in choice
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert choice["finish_reason"] == "stop"

    # Usage
    assert "usage" in data
    assert "prompt_tokens" in data["usage"]
    assert "completion_tokens" in data["usage"]
    assert "total_tokens" in data["usage"]

    print(f"  All required fields present")
    print(f"  ID: {data['id']}, created: {data['created']}")


if __name__ == "__main__":
    results = []

    # Fast tests first (no API calls)
    results.append(test("Health check", test_health))
    results.append(test("List models", test_list_models))
    results.append(test("Get model", test_get_model))

    # API tests (require live connection)
    results.append(test("Basic chat", test_chat_basic))
    results.append(test("System prompt", test_chat_system_prompt))
    results.append(test("Streaming", test_chat_streaming))
    results.append(test("Model alias (gpt-4)", test_chat_model_alias))
    results.append(test("Response structure", test_response_structure))

    passed = sum(results)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"RESULTS: {passed}/{total} passed")
    print(f"{'='*60}")
