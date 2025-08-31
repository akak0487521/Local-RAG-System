"""LLM backend wrappers with fallback and streaming support."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from .config import (
    MODEL_PRIORITY,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_USE_CHAT,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)

logger = logging.getLogger(__name__)

# Cache the OpenAI client so repeated calls do not re-init
_openai_client = None


class OpenAIBackend:
    """Wrapper around OpenAI chat completions."""

    def __init__(self) -> None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")
        global _openai_client
        if _openai_client is None:
            from openai import OpenAI

            _openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.client = _openai_client

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **_: Any,
    ) -> Iterable[str] | str:
        opts: Dict[str, Any] = {"temperature": temperature}
        if isinstance(max_tokens, int) and max_tokens > 0:
            opts["max_tokens"] = max_tokens
        if stream:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL, messages=messages, stream=True, **opts
            )
            for chunk in response:
                delta = ""
                try:
                    delta = getattr(chunk.choices[0].delta, "content", None) or chunk.choices[0].delta.get(
                        "content", ""
                    )
                except Exception:
                    pass
                if delta:
                    yield delta
        else:
            resp = self.client.chat.completions.create(
                model=OPENAI_MODEL, messages=messages, stream=False, **opts
            )
            return resp.choices[0].message.content or ""


class OllamaBackend:
    """Wrapper around the Ollama HTTP API."""

    def __init__(self) -> None:
        if not OLLAMA_MODEL:
            raise RuntimeError("OLLAMA_MODEL not set")
        self.use_chat = OLLAMA_USE_CHAT

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        stream: bool = False,
        temperature: float = 0.7,
        num_predict: Optional[int] = None,
        **_: Any,
    ) -> Iterable[str] | str:
        options: Dict[str, Any] = {"temperature": temperature}
        if isinstance(num_predict, int) and num_predict > 0:
            options["num_predict"] = num_predict
        if self.use_chat:
            payload: Dict[str, Any] = {
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": stream,
                "options": options,
            }
            url = f"{OLLAMA_HOST}/api/chat"
            try:
                if stream:
                    with requests.post(url, json=payload, stream=True, timeout=300) as r:
                        r.raise_for_status()
                        for line in r.iter_lines():
                            if not line:
                                continue
                            data = json.loads(line.decode("utf-8"))
                            msg = (data.get("message") or {}).get("content") or data.get("response")
                            if msg:
                                yield msg
                        return
                else:
                    r = requests.post(url, json=payload, timeout=300)
                    r.raise_for_status()
                    data = r.json()
                    return (data.get("message") or {}).get("content") or data.get("response", "")
            except requests.HTTPError as e:
                if getattr(e.response, "status_code", None) == 404:
                    logger.info("/api/chat not found, falling back to /api/generate")
                    self.use_chat = False
                else:
                    raise

        prompt = "\n".join(m.get("content", "") for m in messages)
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": stream,
            "options": options,
        }
        url = f"{OLLAMA_HOST}/api/generate"
        if stream:
            with requests.post(url, json=payload, stream=True, timeout=300) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line.decode("utf-8"))
                    msg = data.get("response")
                    if msg:
                        yield msg
        else:
            r = requests.post(url, json=payload, timeout=300)
            r.raise_for_status()
            data = r.json()
            return data.get("response", "")


BACKENDS = {
    "openai": OpenAIBackend,
    "ollama": OllamaBackend,
}


def generate(
    messages: List[Dict[str, str]],
    *,
    stream: bool = False,
    engine: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[Iterable[str] | str, str]:
    """Generate a response using the configured backends.

    Returns a tuple of (result, backend_name). ``result`` is either the full
    string response when ``stream`` is False, or an iterator yielding strings
    when ``stream`` is True.
    """

    priorities = [engine] if engine else MODEL_PRIORITY
    last_err: Optional[Exception] = None
    for name in priorities:
        backend_cls = BACKENDS.get(name)
        if not backend_cls:
            continue
        try:
            backend = backend_cls()
        except Exception as e:
            last_err = e
            logger.warning("Backend %s unavailable: %s", name, e)
            continue
        try:
            return backend.generate(messages, stream=stream, **kwargs), name
        except Exception as e:
            last_err = e
            logger.warning("Backend %s failed: %s", name, e)
            continue
    raise RuntimeError("All backends failed") from last_err
