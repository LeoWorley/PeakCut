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
NO_KEY = {"ollama", "custom", "codex_cli"}  # local models, local gateways and the signed-in Codex CLI need no API key

# ---------------------------------------------------------------- the settings in use
_lock = threading.Lock()
_saved = {"at": None, "data": {}}
_codex_lock = threading.Lock()
_codex_cached = {"at": 0.0, "binary": None, "env": {}, "status": None}


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
    k = str(d["key"]) if d.get("provider") and "key" in d else config.ai_key()
    if p == "codex_cli":
        k = ""
    return {"provider": p,
            "models": list(d["models"]) if d.get("provider") else list(config.AI_MODELS),
            "base_url": (str(d.get("base_url") or "") if d.get("provider") else config.AI_BASE_URL).rstrip("/"),
            "key": k}


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
    out = {"provider": c["provider"], "models": c["models"], "base_url": c["base_url"], "has_key": bool(c["key"]),
            "key_hint": c["key"][-4:] if len(c["key"]) >= 12 else "", "ready": ready(), "label": label(),
            "saved_here": bool(_from_dashboard().get("provider")),
            "providers": [{"id": k, "label": v[0], "address": v[1], "needs_key": k not in NO_KEY, "key_url": v[3], "example": v[4]}
                          for k, v in PROVIDERS.items()]}
    out["codex"] = codex_status() if c["provider"] == "codex_cli" else None
    return out


def save(provider_id: str, model_ids, address: str = "", new_key=None) -> dict:
    """Save the choice made in the dashboard. `new_key` None keeps the key that is already there (the page never
    has it to send back); an empty string removes it."""
    p = ALIASES.get(str(provider_id or "").strip().lower(), str(provider_id or "").strip().lower())
    if p not in PROVIDERS:
        raise ValueError("Choose one of the AI services in the list.")
    if isinstance(model_ids, str):
        model_ids = model_ids.split(",")
    ms = [str(m).strip()[:120] for m in (model_ids or []) if str(m).strip()][:5]
    if p == "codex_cli" and not ms:
        ms = ["default"]
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
    k = "" if p == "codex_cli" else (old["key"] if new_key is None and old["provider"] == p else str(new_key or "").strip()[:400])
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
        _codex_cached.update(at=0.0, status=None)
    return public()


def list_models(provider_id=None, address=None, with_key=None) -> list:
    """The model ids a service offers, so the settings page can show a list to pick from."""
    c = current()
    p = ALIASES.get(str(provider_id or c["provider"]).lower(), str(provider_id or c["provider"]).lower())
    if p not in PROVIDERS:
        raise ValueError("Choose one of the AI services in the list.")
    if p == "codex_cli":
        binary, env = _codex_context()
        if not binary:
            raise AIError("Codex CLI was not found. Install it and make sure `codex` is available in your shell.")
        try:
            result = subprocess.run([binary, "debug", "models"], capture_output=True, text=True,
                                    timeout=30, cwd=tempfile.gettempdir(), env=env)
            if result.returncode != 0:
                result = subprocess.run([binary, "debug", "models", "--bundled"], capture_output=True, text=True,
                                        timeout=30, cwd=tempfile.gettempdir(), env=env)
        except subprocess.TimeoutExpired as e:
            raise AIError("Timed out while asking Codex CLI for its model list.") from e
        if result.returncode != 0:
            raise AIError("Codex CLI could not list models. Check `codex login status` and try again.")
        try:
            data = json.loads(result.stdout)
        except ValueError as e:
            raise AIError("Codex CLI returned an unreadable model list.") from e
        found = {str(m.get("slug", "")).strip() for m in data.get("models", [])
                 if isinstance(m, dict) and m.get("visibility", "list") == "list"}
        return ["default", *sorted(m for m in found if m and m != "default")][:400]
    k = str(with_key or "").strip() or (c["key"] if c["provider"] == p else "")
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
            status = codex_status()
            raise ValueError(status["error"] or "Choose a Codex model and log in to Codex CLI first.")
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
    if provider() == "codex_cli":
        if "not found" in low or "not installed" in low:
            return "Codex CLI was not found. Install it, then restart PeakCut."
        if "not logged in" in low or "not authenticated" in low or "login status" in low:
            return "Codex CLI is installed but not logged in. Run `codex login` in Terminal, then test again."
        if "model" in low and ("unknown" in low or "not found" in low):
            return f"Codex does not recognize the model \"{name}\". Press Find models and choose an available model."
        if "429" in text or "rate limit" in low or "usage limit" in low or "quota" in low:
            return "Codex reports a usage limit. Check your Codex account or try again later."
        if "timed out" in low or "timeout" in low:
            return "Codex CLI timed out. Check that it is logged in and try again."
        return "Codex CLI could not complete the request. Check `codex login status` and the selected model."
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


def _codex_context() -> tuple[str | None, dict]:
    """Find Codex even when PeakCut was started by a macOS GUI with a short PATH."""
    cached = _codex_cached
    if cached.get("binary"):
        return cached["binary"], _codex_runtime_env(cached["env"])

    shell_env = {}
    shells = []
    if os.environ.get("SHELL"):
        shells.append(os.environ["SHELL"])
    shells.extend(["/bin/zsh", "/bin/bash"])
    for shell in dict.fromkeys(shells):
        if not os.path.isfile(shell):
            continue
        try:
            result = subprocess.run([shell, "-ilc", "env -0"], capture_output=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode != 0:
            continue
        for entry in result.stdout.split(b"\0"):
            key_name, sep, value = entry.partition(b"=")
            if sep:
                shell_env[key_name.decode(errors="replace")] = value.decode(errors="replace")
        if shell_env.get("PATH"):
            break

    env = os.environ.copy()
    if shell_env.get("PATH"):
        env["PATH"] = shell_env["PATH"]
    if shell_env.get("CODEX_HOME") and not env.get("CODEX_HOME"):
        env["CODEX_HOME"] = shell_env["CODEX_HOME"]
    binary = shutil.which("codex", path=os.environ.get("PATH", "")) or shutil.which("codex", path=env.get("PATH", ""))
    with _codex_lock:
        _codex_cached.update(binary=binary, env={k: env[k] for k in ("PATH", "CODEX_HOME") if env.get(k)})
    return binary, _codex_runtime_env(_codex_cached["env"])


def _codex_runtime_env(overrides: dict) -> dict:
    """Keep the local CLI login, not an unrelated API-key environment variable."""
    env = os.environ.copy()
    env.update(overrides)
    env.pop("OPENAI_API_KEY", None)
    env.pop("CODEX_API_KEY", None)
    return env


def codex_status(refresh: bool = False) -> dict:
    """Return installed/login status, cached briefly because status endpoints poll frequently."""
    now = time.monotonic()
    with _codex_lock:
        if not refresh and _codex_cached["status"] is not None and now - _codex_cached["at"] < 20:
            return dict(_codex_cached["status"])
    binary, env = _codex_context()
    if not binary:
        result = {"installed": False, "authenticated": False, "error": "Codex CLI was not found on PATH."}
    else:
        try:
            proc = subprocess.run([binary, "login", "status"], capture_output=True, text=True,
                                  timeout=15, cwd=tempfile.gettempdir(), env=env)
            output = (proc.stdout + "\n" + proc.stderr).strip()
            authenticated = proc.returncode == 0 and "logged in" in output.lower() and "not logged in" not in output.lower()
            result = {"installed": True, "authenticated": authenticated,
                      "error": None if authenticated else ("Codex CLI is installed but not logged in. Run `codex login` in Terminal." if "not logged in" in output.lower() or not output else "Could not verify Codex CLI login status.")}
        except subprocess.TimeoutExpired:
            result = {"installed": True, "authenticated": False, "error": "Timed out checking Codex CLI login status."}
        except OSError:
            result = {"installed": False, "authenticated": False, "error": "Codex CLI could not be started."}
    with _codex_lock:
        _codex_cached.update(at=now, status=result)
    return dict(result)


def _codex_cli(model: str, prompt: str, schema: dict):
    """Use the logged-in Codex CLI for one schema-constrained classification request."""
    binary, env = _codex_context()
    if not binary:
        raise AIError("Codex CLI was not found. Install it and make sure `codex` is available in your shell.")
    status = codex_status()
    if not status["authenticated"]:
        raise AIError(status["error"] or "Codex CLI is not logged in.")

    with tempfile.TemporaryDirectory(prefix="peakcut-codex-") as workdir:
        schema_path = os.path.join(workdir, "schema.json")
        output_schema = {"type": "OBJECT", "properties": {"items": _codex_strict_schema(schema)},
                         "required": ["items"], "additionalProperties": False}
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(_json_schema(output_schema), f)
        args = [binary, "exec", "--ephemeral", "--ignore-user-config", "--sandbox", "read-only",
                "--ask-for-approval", "never", "--skip-git-repo-check", "--json", "--output-schema", schema_path]
        if model and model != "default":
            args.extend(["--model", model])
        args.append("-")
        try:
            proc = subprocess.run(args, input=_wrapped(prompt, schema), capture_output=True, text=True, timeout=180,
                                  cwd=workdir, env=env)
        except subprocess.TimeoutExpired as e:
            raise AIError("Codex CLI timed out after 180 seconds.") from e
        except OSError as e:
            raise AIError(f"Codex CLI could not be started: {e}") from e

    usage_info, answer, cli_error = {}, "", ""
    for line in proc.stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "turn.completed":
            usage_info = event.get("usage") or {}
        elif event.get("type") == "turn.failed":
            cli_error = str((event.get("error") or {}).get("message") or "")
        elif event.get("type") == "item.completed":
            item = event.get("item") or {}
            if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                answer = item["text"]
    if proc.returncode != 0:
        detail = cli_error or proc.stderr.strip()
        detail = detail[:300] if detail else f"exit code {proc.returncode}"
        for secret_name in ("OPENAI_API_KEY", "CODEX_API_KEY"):
            secret = env.get(secret_name, "")
            if secret:
                detail = detail.replace(secret, "<redacted>")
        raise AIError(f"Codex CLI failed: {detail}")
    if not answer:
        raise AIError("Codex CLI finished without a final structured response.")
    return _parse(answer), int(usage_info.get("input_tokens") or 0), int(usage_info.get("output_tokens") or 0)


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


def _codex_strict_schema(schema):
    """Make PeakCut's schemas valid for Codex structured output, which is strict."""
    def nullable(s):
        out = dict(s)
        kind = out.get("type")
        if "enum" in out and None not in out["enum"]:
            out["enum"] = [*out["enum"], None]
        if isinstance(kind, str):
            out["type"] = [kind, "null"]
        elif isinstance(kind, list) and "null" not in kind:
            out["type"] = [*kind, "null"]
        elif kind is None and "enum" not in out:
            out = {"anyOf": [out, {"type": "null"}]}
        return out

    def strict(node):
        if isinstance(node, list):
            return [strict(item) for item in node]
        if not isinstance(node, dict):
            return node
        out = {key: strict(value) for key, value in node.items()}
        if out.get("type") == "object":
            properties = out.setdefault("properties", {})
            required = set(out.get("required", []))
            for name, prop in properties.items():
                if name not in required:
                    properties[name] = nullable(prop)
            out["required"] = list(properties)
            out["additionalProperties"] = False
        return out

    return strict(_json_schema(schema))


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


# ---------------------------------------------------------------- one call
def call(model: str, prompt: str, schema: dict):
    """Ask one model. Returns (the parsed JSON answer, tokens sent, tokens written back)."""
    if provider() == "codex_cli":
        return _codex_cli(model, prompt, schema)
    how = PROVIDERS[provider()][2]
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
