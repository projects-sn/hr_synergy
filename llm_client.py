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
