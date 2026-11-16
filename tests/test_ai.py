"""The AI layer, with the network replaced: every kind of service is asked correctly and its answer is read
correctly. Nothing leaves this machine and no key is needed."""
import json
import os
import sys
import tempfile
from types import SimpleNamespace

os.environ["PEAKCUT_DATA"] = tempfile.mkdtemp(prefix="pc-ai-")
os.environ["DATABASE_URL"] = ""
os.environ.update(PEAKCUT_AI_PROVIDER="gemini", PEAKCUT_AI_MODELS="model-a, model-b", PEAKCUT_AI_KEY="test-key-123456", PEAKCUT_AI_BASE_URL="")
sys.path.insert(0, ".")
from studio import ai, auth, brain, config, ctx, db, usage  # noqa: E402

db.init()
fails = []


def check(name, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {extra}" if extra and not ok else ""))
    ok or fails.append(name)


class Reply:
    def __init__(self, status, data):
        self.status_code, self._data, self.text = status, data, json.dumps(data)

    def json(self):
        return self._data


CLIP = {"start": 1, "end": 20, "score": 80, "hook": "WOW", "title": "t", "description": "d", "hashtags": ["a"], "reason": "r"}
seen = []


def fake_post(url, json=None, headers=None, timeout=None, **_k):  # noqa: A002
    seen.append((url, json, headers or {}))
    if url.endswith(":generateContent"):
        return Reply(200, {"candidates": [{"content": {"parts": [{"text": globals()["json_text"]([CLIP])}]}}],
                           "usageMetadata": {"promptTokenCount": 100, "totalTokenCount": 130}})
    if url.endswith("/chat/completions"):
        if "temperature" in json:                      # a model that refuses an extra setting: asked again without it
            return Reply(400, {"error": {"message": "temperature is not supported"}})
        return Reply(200, {"choices": [{"message": {"content": "```json\n" + globals()["json_text"]({"items": [CLIP]}) + "\n```"}}],
                           "usage": {"prompt_tokens": 90, "completion_tokens": 20}})
    if url.endswith("/messages"):
        return Reply(200, {"content": [{"type": "text", "text": "Here you go: " + globals()["json_text"]({"items": [CLIP]})}],
                           "usage": {"input_tokens": 70, "output_tokens": 10}})
    return Reply(404, {})


def json_text(x):
    return json.dumps(x)


ai.httpx.post = fake_post


def use(provider, base=""):
    config.AI_PROVIDER, config.AI_BASE_URL = provider, base
    seen.clear()


use("gemini")
out, sent, wrote = ai.call("model-a", "hello", brain.SCHEMA)
check("gemini: answer and tokens", out == [CLIP] and (sent, wrote) == (100, 30), str((out, sent, wrote)))
check("gemini: key in the header, schema passed as is", seen[0][2].get("x-goog-api-key") == "test-key-123456"
      and seen[0][1]["generationConfig"]["responseSchema"] is brain.SCHEMA and seen[0][0].endswith("/model-a:generateContent"))

for name, host in (("openai", "api.openai.com"), ("xai", "api.x.ai"), ("grok", "api.x.ai"), ("groq", "api.groq.com"), ("openrouter", "openrouter.ai")):
    use(name)
    out, sent, wrote = ai.call("model-a", "hello", brain.SCHEMA)
    check(f"{name}: answer unwrapped from a code fence, tokens read", out == [CLIP] and (sent, wrote) == (90, 20), str((out, sent, wrote)))
    check(f"{name}: right address, key as a bearer token", host in seen[0][0] and seen[0][2].get("Authorization") == "Bearer test-key-123456")
check("a refused extra setting is dropped and asked again", len(seen) == 2 and "temperature" not in seen[1][1])
check("the schema is told in the prompt for chat models", '"type": "array"' in seen[1][1]["messages"][0]["content"])

use("anthropic")
out, sent, wrote = ai.call("model-a", "hello", brain.SCHEMA)
check("anthropic: answer found inside a sentence, tokens read", out == [CLIP] and (sent, wrote) == (70, 10), str((out, sent, wrote)))
check("anthropic: key and version headers", seen[0][2].get("x-api-key") == "test-key-123456" and "anthropic-version" in seen[0][2])

use("custom", "http://127.0.0.1:1234/v1")
out, _s, _w = ai.call("model-a", "hello", brain.SCHEMA)
check("custom: uses the address from settings", seen[0][0] == "http://127.0.0.1:1234/v1/chat/completions" and out == [CLIP])

use("custom")
check("custom without an address is not ready", not ai.ready())
use("nonsense")
check("an unknown provider is not ready", not ai.ready())
use("ollama")
os.environ["PEAKCUT_AI_KEY"] = ""
config._kept.clear()
check("ollama needs no key", ai.ready() and ai.label().startswith("Ollama"))
out, _s, _w = ai.call("model-a", "hello", brain.SCHEMA)
check("ollama: no key header is sent", "Authorization" not in seen[0][2] and out == [CLIP])
use("gemini")
check("gemini without a key is not ready", not ai.ready())
os.environ["PEAKCUT_AI_KEY"] = "test-key-123456"
config._kept.clear()

# ---- Codex CLI: use the logged-in account, structured output and its model catalog
use("codex_cli")
ai._codex_cached.update(binary="/fake/codex", env={"PATH": os.environ.get("PATH", "")},
                        at=__import__("time").monotonic(),
                        status={"installed": True, "authenticated": True, "error": None})
real_run = ai.subprocess.run
codex_seen = []
schemas_seen = []


def fake_codex_run(args, **kwargs):
    codex_seen.append((args, kwargs))
    if args[1:3] == ["debug", "models"]:
        return SimpleNamespace(returncode=0, stdout=json.dumps({"models": [
            {"slug": "gpt-6-sol", "visibility": "list"},
            {"slug": "hidden-model", "visibility": "hide"},
        ]}), stderr="")
    if "exec" in args:
        schema_path = args[args.index("--output-schema") + 1]
        with open(schema_path, encoding="utf-8") as f:
            schema_json = json.load(f)
        schemas_seen.append(schema_json)
        events = [
            {"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps({"items": [CLIP]})}},
            {"type": "turn.completed", "usage": {"input_tokens": 31, "output_tokens": 12}},
        ]
        return SimpleNamespace(returncode=0, stdout="\n".join(json.dumps(x) for x in events), stderr="",
                               schema_json=schema_json)
    return SimpleNamespace(returncode=0, stdout="Logged in using ChatGPT", stderr="")


ai.subprocess.run = fake_codex_run
try:
    check("Codex needs no API key", ai.key() == "" and ai.ready())
    check("Codex model list comes from its CLI catalog", ai.list_models("codex_cli") == ["default", "gpt-6-sol"])
    out, sent, wrote = ai.call("gpt-6-sol", "hello", brain.SCHEMA)
    exec_call = codex_seen[-1]
    check("Codex CLI returns structured data and token usage", out == [CLIP] and (sent, wrote) == (31, 12))
    check("Codex uses isolated read-only ephemeral execution", "--ephemeral" in exec_call[0] and
          exec_call[0][exec_call[0].index("--sandbox") + 1] == "read-only" and
          exec_call[0][exec_call[0].index("--output-schema") + 1].endswith("schema.json") and
          exec_call[1]["cwd"].startswith(tempfile.gettempdir()))
    check("Codex wraps array schemas in a standard JSON Schema object", schemas_seen[0]["type"] == "object" and
          schemas_seen[0]["properties"]["items"]["type"] == "array" and
          exec_call[1]["input"].startswith("hello") and exec_call[0][-1] == "-" and exec_call[1]["text"] is True)
    ai.call("default", "hello", brain.FIX_SCHEMA)
    check("Codex default model is selected without --model", "--model" not in codex_seen[-1][0])
    ai._codex_cached["status"] = None
    ai.subprocess.run = lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="Not logged in", stderr="")
    status = ai.codex_status(refresh=True)
    check("Codex login status explains when CLI is signed out", not status["authenticated"] and "codex login" in status["error"])
finally:
    ai.subprocess.run = real_run
    ai._codex_cached.update(at=0.0, status=None, binary=None, env={})
use("gemini")

# ---- the whole path: a call is made for a user and shows up in their usage, and only theirs
u1, _ = auth._create("ai-one@example.com", "One", password_hash="x")
u2, _ = auth._create("ai-two@example.com", "Two", password_hash="x")
words = [{"w": f"word{i}", "s": i * 0.5, "e": i * 0.5 + 0.4} for i in range(120)]
with ctx.as_user(u1.id):
    usage.set_context(None, "A test video")
    import numpy as np
    clips, info = brain.find_clips(words, np.zeros(60), 60.0, title="T", uploader="U")
    check("clips come back through the whole path", len(clips) == 1 and clips[0]["hook"] == "WOW" and info["model"] == "model-a", str((clips, info)))
    s = usage.summary({})
    check("the call is in this user's usage", s["calls_total"] == 1 and s["tokens_total"] == 130 and s["recent"][0]["where"] == "A test video", str(s))
    check("usage says which AI is set", s["ready"] and s["provider"] == "Gemini" and s["model"] == "model-a")
with ctx.as_user(u2.id):
    check("another user sees none of it", usage.summary({})["calls_total"] == 0)

print("ALL PASSED" if not fails else f"{len(fails)} failed")
import shutil  # noqa: E402
db.engine.dispose()
shutil.rmtree(os.environ["PEAKCUT_DATA"], ignore_errors=True)
sys.exit(1 if fails else 0)
