from __future__ import annotations

import os
import orjson
from typing import Any, Dict, List

from openai import OpenAI


def get_openai_client() -> OpenAI:
	api_key = "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"
	base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
	if not api_key:
		raise RuntimeError("OPENAI_API_KEY is not set")
	return OpenAI(api_key=api_key, base_url=base_url)


def chat_json(
	messages: List[Dict[str, Any]],
	model: str,
	temperature: float = 0.0,
	max_tokens: int | None = None,
) -> Dict[str, Any]:
	"""Call Chat Completions with JSON output mode."""
	client = get_openai_client()
	resp = client.chat.completions.create(
		model=model,
		temperature=temperature,
		messages=messages,
		response_format={"type": "json_object"},
		max_tokens=max_tokens,
	)
	content = resp.choices[0].message.content or "{}"
	return orjson.loads(content)


def chat_text(
	messages: List[Dict[str, Any]],
	model: str,
	temperature: float = 0.2,
	max_tokens: int | None = None,
) -> str:
	client = get_openai_client()
	resp = client.chat.completions.create(
		model=model,
		temperature=temperature,
		messages=messages,
		max_tokens=max_tokens,
	)
	return resp.choices[0].message.content or ""
