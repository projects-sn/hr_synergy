from __future__ import annotations

import os
from typing import Any, Dict, Optional

from llm_client import chat_json


DEFAULT_SALARY_MODEL = os.getenv("SALARY_MODEL", os.getenv("ANALYZER_MODEL", "gpt-4o-mini"))


def estimate_salary_rub(
	role_title: str,
	city: str,
	seniority: Optional[str] = None,
	resume_summary: Optional[str] = None,
	job_description: Optional[str] = None,
	model: str = DEFAULT_SALARY_MODEL,
	temperature: float = 0.1,
) -> Dict[str, Any]:
	"""Return a structured estimation in RUB/month using LLM only (no web search)."""
	system_prompt = (
		"Ты – HR-аналитик рынка труда в РФ. На основе входных данных оцени вилку зарплат в рублях в месяц. "
		"Возвращай строго JSON. Не выдумывай фактов, явно указывай неопределённость."
	)

	user_prompt = f"""
Роль/должность: {role_title}
Старшинство: {seniority or 'не указано'}
Город/Локация: {city or 'не указано'}

Краткое резюме кандидата: {resume_summary or '—'}

Описание вакансии (если есть): {job_description or '—'}

Задача: верни JSON с полями:
{{
  "estimate_rub_month": {{"min": int, "max": int, "median": int}},
  "confidence": "low|medium|high",
  "assumptions": ["строка"],
  "sources": ["строка"],
  "notes": "краткие примечания о рынке и допущениях"
}}
"""

	messages = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": user_prompt},
	]

	resp = chat_json(messages=messages, model=model, temperature=temperature)
	return resp


def estimate_salary_from_resume(
	resume_text: str,
	job_description: Optional[str] = None,
	model: str = DEFAULT_SALARY_MODEL,
	temperature: float = 0.1,
) -> Dict[str, Any]:
	"""Infer suitable roles/directions and estimate salary ranges from resume text.

	Returns JSON with:
	{
	  "roles": [{"title": str, "direction": str, "seniority": str|null, "fit_reason": str}],
	  "ranges_per_role": [{"title": str, "min": int, "max": int, "median": int}],
	  "estimate_rub_month": {"min": int, "max": int, "median": int},
	  "confidence": "low|medium|high",
	  "assumptions": [str],
	  "notes": str
	}
	"""
	if not resume_text.strip():
		raise ValueError("resume_text is required")

	system_prompt = (
		"Ты – HR-аналитик рынка труда в РФ. По тексту резюме (и при наличии JD) "
		"определи 3–5 подходящих направлений/профессий и оцени вилку зарплат в рублях/мес. "
		"Возвращай строго JSON как в задаче. Не выдумывай фактов и явно указывай неопределённость."
	)

	user_prompt = f"""
Текст резюме:
{resume_text[:8000]}

Описание вакансии (если есть):
{job_description or '—'}

Задача: верни строго JSON с полями:
{{
  "roles": [{{"title": "строка", "direction": "строка", "seniority": "Junior|Middle|Senior|Lead|null", "fit_reason": "кратко"}}],
  "ranges_per_role": [{{"title": "строка", "min": int, "max": int, "median": int}}],
  "estimate_rub_month": {{"min": int, "max": int, "median": int}},
  "confidence": "low|medium|high",
  "assumptions": ["строка"],
  "notes": "краткие примечания"
}}

Требования:
- Роли должны отражать ключевые навыки и опыт из резюме (и JD, если есть).
- Диапазоны зарплат — реалистичные на текущем рынке РФ.
- Если данных недостаточно — укажи это в notes и повысь неопределённость.
"""

	messages = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": user_prompt},
	]

	resp = chat_json(messages=messages, model=model, temperature=temperature)
	return resp