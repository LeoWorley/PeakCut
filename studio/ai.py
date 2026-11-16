"""Talking to AI providers. Choose the service and model in the dashboard
(Settings, AI) or in .env. What is saved from the dashboard is kept in ai.json in the data folder and wins.

    PEAKCUT_AI_PROVIDER   gemini | openai | xai | groq | openrouter | anthropic | ollama | custom | codex_cli
    PEAKCUT_AI_MODELS     one or more model ids, comma separated, tried in this order
    PEAKCUT_AI_KEY        your key (or PEAKCUT_AI_KEY_FILE, a file holding it)
    PEAKCUT_AI_BASE_URL   only for "custom" (any service that speaks the OpenAI chat format), or to override

Every service is asked for the same thing: JSON that follows a small schema. Gemini takes the schema directly.
The others are told the schema in the prompt and asked for a JSON object. Codex CLI uses its saved local login."""
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import httpx

from . import config

# name -> (label shown on the pages, address, how it is spoken to, where to get a key, an example model id)
PROVIDERS = {
    "gemini": ("Gemini", "https://generativelanguage.googleapis.com/v1beta/models", "gemini", "https://aistudio.google.com/apikey", "gemini-2.5-flash"),
    "openai": ("OpenAI", "https://api.openai.com/v1", "openai", "https://platform.openai.com/api-keys", "gpt-4o-mini"),
    "xai": ("Grok", "https://api.x.ai/v1", "openai", "https://console.x.ai", "grok-4"),
    "groq": ("Groq", "https://api.groq.com/openai/v1", "openai", "https://console.groq.com/keys", "llama-3.3-70b-versatile"),
    "openrouter": ("OpenRouter", "https://openrouter.ai/api/v1", "openai", "https://openrouter.ai/keys", "google/gemini-2.5-flash"),
    "anthropic": ("Claude", "https://api.anthropic.com/v1", "anthropic", "https://console.anthropic.com", ""),
    "ollama": ("Ollama (on this machine)", "http://127.0.0.1:11434/v1", "openai", "https://ollama.com", "llama3.1"),
    "custom": ("Custom (OpenAI-compatible)", "", "openai", "", ""),
    "codex_cli": ("Codex CLI", "", "codex_cli", "", "default"),
}
ALIASES = {"google": "gemini", "grok": "xai", "claude": "anthropic", "local": "ollama"}
NO_KEY = {"ollama", "custom", "codex_cli"}  # local models, gateways and the signed-in Codex CLI need no key

# ---------------------------------------------------------------- the settings in use
_lock = threading.Lock()
_saved = {"at": None, "data": {}}
_codex_lock = threading.Lock()
_codex_cached = {"at": 0.0, "binary": None, "env": {}, "status": None}


def codex_status(refresh=False) -> dict:
    """Whether Codex CLI is installed and its existing local login can be used."""
    now = time.monotonic()
    with _codex_lock:
        if not refresh and _codex_cached["status"] is not None and now - _codex_cached["at"] < 30:
            return dict(_codex_cached["status"])
        binary = shutil.which("codex")
        if not binary:
            status = {"installed": False, "authenticated": False,
                      "error": "Codex CLI was not found. Install it and restart PeakCut."}
        else:
            _codex_cached["binary"] = binary
            try:
                result = subprocess.run([binary, "login", "status"], capture_output=True, text=True,
                                        timeout=10, check=False)
                authenticated = result.returncode == 0
                status = {"installed": True, "authenticated": authenticated,
                          "error": None if authenticated else "Run `codex login` in Terminal, then test again."}
            except (OSError, subprocess.SubprocessError):
                status = {"installed": True, "authenticated": False,
                          "error": "Could not check the Codex CLI login. Run `codex login` in Terminal, then test again."}
        _codex_cached.update(at=now, status=status)
        return dict(status)


def _codex_env(home=None) -> dict:
    """Make an environment that uses the saved login but does not load the user's Codex config."""
    env = os.environ.copy()
    if home is not None:
        isolated = home / "codex-home"
        isolated.mkdir(parents=True, exist_ok=True)
        source = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex") / "auth.json"
        if source.is_file():
            target = isolated / "auth.json"
            try:
                target.symlink_to(source)
            except OSError:
                shutil.copyfile(source, target)
        env["CODEX_HOME"] = str(isolated)
    env["NO_COLOR"] = "1"
    env["TERM"] = "dumb"
    return env


def _file():
    return config.DATA_DIR / "ai.json"


def _from_dashboard() -> dict:
    """What was saved from the dashboard, read again only when the file changes."""
    f = _file()
    try:
        at = f.stat().st_mtime_ns
    except OSError:
        at = 0
    if _saved["at"] != at:
        try:
            data = json.loads(f.read_text(encoding="utf-8")) if at else {}
        except (OSError, ValueError):
            data = {}
        _saved.update(at=at, data=data if isinstance(data, dict) else {})
    return _saved["data"]


def current() -> dict:
    """Provider, models, address and key: what the dashboard saved, or else what .env says."""
    d = _from_dashboard()
    p = str(d.get("provider") or config.AI_PROVIDER or "").strip().lower()
    p = ALIASES.get(p, p)
    return {"provider": p,
            "models": list(d["models"]) if d.get("provider") else list(config.AI_MODELS),
            "base_url": (str(d.get("base_url") or "") if d.get("provider") else config.AI_BASE_URL).rstrip("/"),
            "key": "" if p == "codex_cli" else (str(d["key"]) if d.get("provider") and "key" in d else config.ai_key())}


def provider() -> str:
    return current()["provider"]


def models() -> list:
    return current()["models"]


def key() -> str:
    return current()["key"]


def label() -> str:
    return PROVIDERS.get(provider(), ("",))[0]


def base_url() -> str:
    c = current()
    return c["base_url"] or PROVIDERS.get(c["provider"], ("", ""))[1]


def ready() -> bool:
    c = current()
    if c["provider"] == "codex_cli":
        status = codex_status()
        return bool(c["models"] and status["installed"] and status["authenticated"])
    return bool(c["provider"] in PROVIDERS and c["models"] and base_url() and (c["key"] or c["provider"] in NO_KEY))


def public() -> dict:
    """What the settings page may see. The key itself never leaves the server: only whether there is one, and
    its last four characters so the owner can tell which key it is."""
    c = current()
    return {"provider": c["provider"], "models": c["models"], "base_url": c["base_url"], "has_key": bool(c["key"]),
            "key_hint": c["key"][-4:] if len(c["key"]) >= 12 else "", "ready": ready(), "label": label(),
            "saved_here": bool(_from_dashboard().get("provider")),
            "codex": codex_status(),
            "providers": [{"id": k, "label": v[0], "address": v[1], "needs_key": k not in NO_KEY, "key_url": v[3], "example": v[4]}
                          for k, v in PROVIDERS.items()]}


def save(provider_id: str, model_ids, address: str = "", new_key=None) -> dict:
    """Save the choice made in the dashboard. `new_key` None keeps the key that is already there (the page never
    has it to send back); an empty string removes it."""
    p = ALIASES.get(str(provider_id or "").strip().lower(), str(provider_id or "").strip().lower())
    if p not in PROVIDERS:
        raise ValueError("Choose one of the AI services in the list.")
    if isinstance(model_ids, str):
        model_ids = model_ids.split(",")
    ms = [str(m).strip()[:120] for m in (model_ids or []) if str(m).strip()][:5]
    if not ms:
        raise ValueError("Enter the model to use.")
    if any(not re.fullmatch(r"[\w.\-:/@]+", m) for m in ms):
        raise ValueError("A model id has letters, digits and . - : / only.")
    address = str(address or "").strip().rstrip("/")[:300]
    if address and not re.fullmatch(r"https?://[^\s]+", address):
        raise ValueError("The address has to start with http:// or https://")
    if p == "custom" and not address:
        raise ValueError("A custom service needs its address, for example http://127.0.0.1:1234/v1")
    old = current()
    k = old["key"] if new_key is None and old["provider"] == p else str(new_key or "").strip()[:400]
    if not k and p not in NO_KEY:
        raise ValueError("Paste the key for this service.")
    with _lock:
        f = _file()
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps({"provider": p, "models": ms, "base_url": address, "key": k}), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)        # the key is only for whoever runs the server
        except OSError:
            pass
        tmp.replace(f)
        _saved["at"] = None
    return public()


def list_models(provider_id=None, address=None, with_key=None) -> list:
    """The model ids a service offers, so the settings page can show a list to pick from."""
    c = current()
    p = ALIASES.get(str(provider_id or c["provider"]).lower(), str(provider_id or c["provider"]).lower())
    if p not in PROVIDERS:
        raise ValueError("Choose one of the AI services in the list.")
    k = str(with_key or "").strip() or (c["key"] if c["provider"] == p else "")
    if p == "codex_cli":
        status = codex_status()
        if not status["installed"] or not status["authenticated"]:
            raise ValueError(status["error"] or "Run `codex login` in Terminal, then try again.")
        binary = _codex_cached.get("binary") or shutil.which("codex")
        if not binary:
            raise ValueError("Codex CLI was not found. Install it and restart PeakCut.")
        with tempfile.TemporaryDirectory(prefix="peakcut-codex-models-") as tmp:
            result = subprocess.run([binary, "debug", "models"], capture_output=True, text=True, timeout=20,
                                    check=False, cwd=tmp, env=_codex_env(Path(tmp)))
        if result.returncode:
            raise AIError((result.stderr or result.stdout or "Codex CLI could not list models.")[:300])
        found = []
        output = (result.stdout or "").strip()
        try:
            data = json.loads(output)
        except ValueError:
            data = None
        if isinstance(data, dict):
            data = data.get("models") or data.get("data") or []
        if isinstance(data, list):
            found.extend(str(m.get("id") or m.get("model") or "") if isinstance(m, dict) else str(m) for m in data)
        if not found:
            for line in output.splitlines():
                match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._:/-]{1,119})\s{2,}", line)
                if match and match.group(1).lower() not in {"model", "models", "name", "id", "default"}:
                    found.append(match.group(1))
        if not found:
            found = re.findall(r"\b(?:gpt-[A-Za-z0-9._-]+|o[1-9][A-Za-z0-9._-]*|codex-[A-Za-z0-9._-]+)\b", output)
        return ["default"] + sorted({m.strip() for m in found if m.strip() and m.strip() != "default"})[:399]
    base = str(address or "").strip().rstrip("/") or (c["base_url"] if c["provider"] == p else "") or PROVIDERS[p][1]
    if not base:
        raise ValueError("Enter the service's address first.")
    how = PROVIDERS[p][2]
    if how == "gemini":
        r = httpx.get(base, params={"pageSize": 200}, headers={"x-goog-api-key": k}, timeout=20)
    elif how == "anthropic":
        r = httpx.get(f"{base}/models", params={"limit": 200}, headers={"x-api-key": k, "anthropic-version": "2023-06-01"}, timeout=20)
    else:
        r = httpx.get(f"{base}/models", headers={"Authorization": f"Bearer {k}"} if k else {}, timeout=20)
    if r.status_code != 200:
        raise AIError(f"HTTP {r.status_code} {r.text[:200].replace(k or chr(0), '<key>')}")
    data = r.json()
    if how == "gemini":
        found = [m.get("name", "").split("/")[-1] for m in data.get("models", [])
                 if "generateContent" in (m.get("supportedGenerationMethods") or [])]
    else:
        found = [m.get("id", "") for m in data.get("data", []) if isinstance(m, dict)]
    return sorted({m for m in found if m})[:400]


def test() -> dict:
    """One tiny request with the settings in use, to prove the key, the address and the model all work."""
    ms = models()
    if not ready():
        if provider() == "codex_cli":
            raise ValueError(codex_status()["error"] or "Run `codex login` in Terminal first.")
        raise ValueError("Save a service, a model and a key first.")
    out, sent, wrote = call(ms[0], 'Answer with a JSON array that holds the single string "ok".', {"type": "ARRAY", "items": {"type": "STRING"}})
    return {"ok": True, "model": ms[0], "tokens": sent + wrote, "answer": str(out)[:80]}


class AIError(RuntimeError):
    pass


problem = {"text": "", "at": 0.0}       # why the AI last failed, for the dashboard; cleared when it answers again


def explain(e) -> str:
    """What went wrong with an AI call, in words the owner can act on."""
    text, low = str(e), str(e).lower()
    name = (models() or ["?"])[0]
    if provider() == "codex_cli" and ("login" in low or "auth" in low or "not logged" in low):
        return "Codex CLI is not logged in. Run `codex login` in Terminal, then test again."
    if "404" in text or "not found" in low or "does not exist" in low or "unknown model" in low or "not_found" in low:
        return f"{label()} doesn't know the model \"{name}\". Open Settings, AI, press Find models and pick one from the list."
    if "401" in text or "403" in text or "api key" in low or "api_key" in low or "unauthorized" in low or "permission" in low:
        return f"{label()} refused the key. Paste a working key under Settings, AI."
    if "429" in text or "quota" in low or "rate limit" in low or "exhausted" in low:
        return f"{label()} says the key has hit its limit or has no credit left. Check the account with {label()}, or try again later."
    if "connect" in low or "timed out" in low or "timeout" in low or "getaddrinfo" in low or "refused" in low:
        return f"PeakCut couldn't reach {label()}. Check the internet connection" + (" and that Ollama is running." if provider() == "ollama" else ".")
    return ""


def note_problem(e):
    import time
    why = explain(e)
    if why:
        problem.update(text=why, at=time.time())


def clear_problem():
    problem.update(text="", at=0.0)


def _fail(r, model):
    text = r.text[:300].replace(key() or "\0", "<key>")
    raise AIError(f"HTTP {r.status_code} {text}")


# ---------------------------------------------------------------- the schema, said in words for chat models
def _json_schema(s):
    """Gemini's schema spelling (ARRAY, OBJECT...) as ordinary JSON Schema."""
    if isinstance(s, dict):
        return {k: (v.lower() if k == "type" and isinstance(v, str) else _json_schema(v)) for k, v in s.items()}
    if isinstance(s, list):
        return [_json_schema(x) for x in s]
    return s


def _wrapped(prompt, schema):
    return (prompt + "\n\nAnswer with JSON only, no other text and no code fences: one object with a single key "
            "\"items\", whose value follows this JSON Schema:\n" + json.dumps(_json_schema(schema)))


def _parse(text):
    """The JSON in an answer, even if the model wrapped it in a code fence or a sentence."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        out = json.loads(text)
    except ValueError:
        m = re.search(r"[\[{].*[\]}]", text, re.S)
        if not m:
            raise AIError("The AI did not answer with JSON.")
        out = json.loads(m.group(0))
    if isinstance(out, dict) and "items" in out:
        out = out["items"]
    return out


def _codex_schema(schema):
    """Convert a chat schema to Codex's strict structured-output JSON Schema."""
    schema = _json_schema(schema)
    if isinstance(schema, dict) and schema.get("type") == "array":
        schema = {"type": "object", "properties": {"items": schema}, "required": ["items"],
                  "additionalProperties": False}

    def strict(node):
        if not isinstance(node, dict):
            return node
        out = dict(node)
        if out.get("type") == "object":
            properties = out.get("properties") or {}
            required = set(out.get("required") or [])
            fixed = {}
            for name, value in properties.items():
                value = strict(value)
                fixed[name] = value if name in required else {"anyOf": [value, {"type": "null"}]}
            out["properties"] = fixed
            out["required"] = list(properties)
            out["additionalProperties"] = False
        elif out.get("type") == "array" and "items" in out:
            out["items"] = strict(out["items"])
        for key in ("anyOf", "oneOf", "allOf"):
            if key in out:
                out[key] = [strict(value) for value in out[key]]
        return out

    return strict(schema)


def _codex(model, prompt, schema):
    status = codex_status()
    if not status["installed"] or not status["authenticated"]:
        raise AIError(status["error"] or "Run `codex login` in Terminal first.")
    binary = _codex_cached.get("binary") or shutil.which("codex")
    if not binary:
        raise AIError("Codex CLI was not found. Install it and restart PeakCut.")
    with tempfile.TemporaryDirectory(prefix="peakcut-codex-") as tmp:
        root = Path(tmp)
        schema_path = root / "schema.json"
        schema_path.write_text(json.dumps(_codex_schema(schema)), encoding="utf-8")
        command = [binary, "exec", "--ephemeral", "--ignore-user-config", "--skip-git-repo-check",
                   "--sandbox", "read-only", "--output-schema", str(schema_path), "--json"]
        if model and model != "default":
            command.extend(["--model", model])
        command.append("-")
        try:
            result = subprocess.run(command, input=prompt, cwd=tmp, capture_output=True, text=True, timeout=180,
                                    check=False, env=_codex_env(root))
        except (OSError, subprocess.SubprocessError) as e:
            raise AIError(f"Could not run Codex CLI: {e}") from e
    if result.returncode:
        message = (result.stderr or result.stdout or "Codex CLI did not complete the request.").strip()[:300]
        raise AIError(message)
    answer, sent, wrote = "", 0, 0
    for line in (result.stdout or "").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "item.completed":
            item = event.get("item") or {}
            if item.get("type") in ("agent_message", "assistant_message"):
                answer = item.get("text") or answer
        elif event.get("type") == "turn.completed":
            usage = event.get("usage") or {}
            sent = int(usage.get("input_tokens") or 0)
            wrote = int(usage.get("output_tokens") or 0)
    if not answer:
        raise AIError("Codex CLI finished without a structured answer.")
    return _parse(answer), sent, wrote


# ---------------------------------------------------------------- one call
def call(model: str, prompt: str, schema: dict):
    """Ask one model. Returns (the parsed JSON answer, tokens sent, tokens written back)."""
    how = PROVIDERS[provider()][2]
    if how == "codex_cli":
        return _codex(model, prompt, schema)
    return {"gemini": _gemini, "openai": _openai, "anthropic": _anthropic}[how](model, prompt, schema)


def _gemini(model, prompt, schema):
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.4, "responseMimeType": "application/json", "responseSchema": schema}}
    r = httpx.post(f"{base_url()}/{model}:generateContent", json=body, headers={"x-goog-api-key": key()}, timeout=120)
    if r.status_code != 200:
        _fail(r, model)
    data = r.json()
    text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
    u = data.get("usageMetadata", {})
    sent = int(u.get("promptTokenCount") or 0)
    return json.loads(text), sent, max(0, int(u.get("totalTokenCount") or 0) - sent)   # the answer, and any thinking billed with it


def _openai(model, prompt, schema):
    headers = {"Authorization": f"Bearer {key()}"} if key() else {}
    body = {"model": model, "messages": [{"role": "user", "content": _wrapped(prompt, schema)}]}
    # Most services accept these two extras; a few models refuse one of them, so they are dropped on a refusal.
    extras = [{"temperature": 0.4, "response_format": {"type": "json_object"}}, {"response_format": {"type": "json_object"}}, {}]
    r = None
    for extra in extras:
        r = httpx.post(f"{base_url()}/chat/completions", json={**body, **extra}, headers=headers, timeout=180)
        if r.status_code not in (400, 422):
            break
    if r.status_code != 200:
        _fail(r, model)
    data = r.json()
    u = data.get("usage") or {}
    return _parse(data["choices"][0]["message"].get("content")), int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)


def _anthropic(model, prompt, schema):
    body = {"model": model, "max_tokens": 8192, "messages": [{"role": "user", "content": _wrapped(prompt, schema)}]}
    r = httpx.post(f"{base_url()}/messages", json=body, timeout=180,
                   headers={"x-api-key": key(), "anthropic-version": "2023-06-01"})
    if r.status_code != 200:
        _fail(r, model)
    data = r.json()
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    u = data.get("usage") or {}
    return _parse(text), int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
