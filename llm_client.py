from __future__ import annotations

import os
import orjson
from typing import Any, Dict, List

from openai import OpenAI
import streamlit as st
import time


def get_openai_client() -> OpenAI:
    """Create OpenAI client using Streamlit secrets or environment variables.

    Precedence: st.secrets > env vars. Fails fast if key is missing.
    """
    api_key = (
        st.secrets.get("OPENAI_API_KEY")
        if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets
        else os.getenv("OPENAI_API_KEY")
    )
    base_url = (
        st.secrets.get("OPENAI_BASE_URL")
        if hasattr(st, "secrets") and "OPENAI_BASE_URL" in st.secrets
        else os.getenv("OPENAI_BASE_URL", "https:\/\/api.openai.com\/v1")
    )
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set (use Streamlit secrets or env var)")
    return OpenAI(api_key=api_key, base_url=base_url)


def chat_json(
	messages: List[Dict[str, Any]],
	model: str,
	temperature: float = 0.0,
	max_tokens: int | None = None,
) -> Dict[str, Any]:
	"""Call Chat Completions with JSON output mode."""
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            client = get_openai_client()
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
                timeout=60.0,
            )
            content = resp.choices[0].message.content or "{}"
            return orjson.loads(content)
        except Exception as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"LLM request failed after retries: {last_err}")


def chat_text(
	messages: List[Dict[str, Any]],
	model: str,
	temperature: float = 0.2,
	max_tokens: int | None = None,
) -> str:
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            client = get_openai_client()
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages,
                max_tokens=max_tokens,
                timeout=60.0,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"LLM request failed after retries: {last_err}")


def get_llm_config() -> Dict[str, Any]:
    """Return resolved LLM config without exposing secrets.

    Includes whether key is present and which base_url is used.
    """
    key_present = bool(
        (hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets)
        or os.getenv("OPENAI_API_KEY")
    )
    base_url = (
        st.secrets.get("OPENAI_BASE_URL")
        if hasattr(st, "secrets") and "OPENAI_BASE_URL" in st.secrets
        else os.getenv("OPENAI_BASE_URL", "https:\/\/api.openai.com\/v1")
    )
    return {"key_present": key_present, "base_url": base_url}


def connectivity_check() -> Dict[str, Any]:
    """Attempt a lightweight API call to verify connectivity."""
    try:
        client = get_openai_client()
        _ = client.models.list()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}
