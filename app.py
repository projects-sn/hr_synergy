from __future__ import annotations

import os
import json
import orjson
import streamlit as st

from prompts import (
	ANALYZER_SYSTEM_PROMPT,
	ANALYZER_USER_TEMPLATE,
	EDITOR_SYSTEM_PROMPT,
	EDITOR_USER_TEMPLATE,
)
from llm_client import chat_json, chat_text
from pdf_utils import extract_text_from_pdf

ANALYZER_MODEL = os.getenv("ANALYZER_MODEL", "gpt-4o-mini")
EDITOR_MODEL = os.getenv("EDITOR_MODEL", "gpt-4o")

st.set_page_config(page_title="Нейро‑HR — анализ и редактура резюме", layout="wide")

st.title("🎯 Нейро‑HR — анализ и редактура резюме")

with st.sidebar:
	st.header("Входные данные")
	resume_pdf = st.file_uploader("Загрузите PDF резюме", type=["pdf"])  # type: ignore
	resume_text_manual = st.text_area("Или вставьте текст резюме", height=180)
	job_description = st.text_area("Описание вакансии", height=180)
	analyzer_temp = st.slider("Температура Анализатор", 0.0, 0.5, 0.1, 0.1)
	editor_temp = st.slider("Температура Редактор", 0.0, 0.6, 0.3, 0.1)


def load_resume_text() -> str:
	if resume_pdf is not None:
		# Save to temp and extract
		tmp_path = os.path.join(st.session_state.get("tmp_dir", "."), "_resume_tmp.pdf")
		with open(tmp_path, "wb") as f:
			f.write(resume_pdf.getbuffer())
		return extract_text_from_pdf(tmp_path)
	return resume_text_manual.strip()


def format_analysis_report(analysis_json: dict) -> str:
	"""Форматирует JSON-отчёт Анализатора в человекочитаемый Markdown"""
	report = []
	
	# Общая оценка
	if "overall_assessment" in analysis_json:
		report.append(f"### Общая оценка\n{analysis_json['overall_assessment']}\n")
	
	# Топ проблем
	if "top_issues" in analysis_json and analysis_json["top_issues"]:
		report.append("### Основные проблемы")
		for i, issue in enumerate(analysis_json["top_issues"], 1):
			severity = issue.get("severity", "medium")
			severity_badge = {
				"high": "**КРИТИЧНО**", 
				"medium": "**ВАЖНО**", 
				"low": "**МЕЛКО**"
			}.get(severity, "**ВАЖНО**")
			
			report.append(f"#### {i}. {severity_badge}")
			report.append(f"**Проблема:** {issue.get('issue', 'Неизвестная проблема')}")
			report.append(f"**Почему это важно:** {issue.get('why', 'Не указано')}")
			report.append(f"**Решение:** {issue.get('fix_suggestion', 'Не указано')}\n")
	
	# Отсутствующие данные
	if "missing_data" in analysis_json and analysis_json["missing_data"]:
		report.append("## Отсутствующие данные")
		for missing in analysis_json["missing_data"]:
			field_name = missing.get('field', 'Неизвестное поле')
			field_display = {
				"metric": "Метрики и результаты",
				"dates": "Даты работы", 
				"location": "Местоположение",
				"education": "Образование",
				"contact": "Контакты",
				"skills": "Навыки"
			}.get(field_name, f"{field_name}")
			report.append(f"- **{field_display}:** {missing.get('note', 'Не указано')}")
		report.append("")
	
	# Соответствие ключевых слов
	if "keywords_match" in analysis_json:
		keywords = analysis_json["keywords_match"]
		report.append("### Соответствие ключевых слов")
		if keywords.get("found_in_resume"):
			report.append(f"**Найдено в резюме:** {', '.join(keywords['found_in_resume'])}")
		if keywords.get("missing"):
			report.append(f"**Отсутствует:** {', '.join(keywords['missing'])}")
		report.append("")
	
	# Риски
	if "risks" in analysis_json and analysis_json["risks"]:
		report.append("### Риски форматирования")
		for risk in analysis_json["risks"]:
			report.append(f"- {risk}")
		report.append("")
	
	# Вопросы кандидату
	if "candidate_questions" in analysis_json and analysis_json["candidate_questions"]:
		report.append("### Вопросы для уточнения")
		for i, question in enumerate(analysis_json["candidate_questions"], 1):
			report.append(f"{i}. {question}")
		report.append("")
	
	# Приоритетный список исправлений
	if "priority_fix_list" in analysis_json and analysis_json["priority_fix_list"]:
		report.append("### План действий")
		for i, fix in enumerate(analysis_json["priority_fix_list"], 1):
			report.append(f"**{i}.** {fix}")
		report.append("")
	
	return "\n".join(report)


st.header("🔹 1) Анализатор")
if st.button("Запустить анализ"):
	resume_text = load_resume_text()
	if not resume_text:
		st.warning("Требуется резюме (PDF или текст)")
	else:
		user_prompt = ANALYZER_USER_TEMPLATE.format(
			resume_text=resume_text,
			job_description=job_description or "",
		)
		messages = [
			{"role": "system", "content": ANALYZER_SYSTEM_PROMPT},
			{"role": "user", "content": user_prompt},
		]
		with st.spinner("Модель анализирует резюме…"):
			try:
				analysis_json = chat_json(
					messages=messages,
					model=ANALYZER_MODEL,
					temperature=float(analyzer_temp),
				)
				st.session_state["analysis_json"] = analysis_json
				st.success("Готово: отчёт сформирован")
			except Exception as e:
				st.error(f"Ошибка LLM: {e}")

# Показываем результаты анализа
if "analysis_json" in st.session_state:
	analysis_json = st.session_state["analysis_json"]
	
	# Создаём табы для разных форматов
	tab1, tab2 = st.tabs(["Анализ резюме", "JSON данные"])
	
	with tab1:
		st.markdown(format_analysis_report(analysis_json))
	
	with tab2:
		st.code(orjson.dumps(analysis_json, option=orjson.OPT_INDENT_2).decode(), language="json")


st.header("🔹 2) Редактор")
if st.button("Сгенерировать улучшенное резюме"):
	resume_text = load_resume_text()
	if not resume_text:
		st.warning("Требуется резюме (PDF или текст)")
	else:
		if "analysis_json" not in st.session_state:
			st.info("Сначала запустите Анализатор — его вывод используется Редактором")
		analysis_json_str = orjson.dumps(st.session_state.get("analysis_json", {})).decode()
		user_prompt = EDITOR_USER_TEMPLATE.format(
			analyzer_json=analysis_json_str,
			resume_text=resume_text,
			job_description=job_description or "",
		)
		messages = [
			{"role": "system", "content": EDITOR_SYSTEM_PROMPT},
			{"role": "user", "content": user_prompt},
		]
		with st.spinner("Модель переписывает резюме…"):
			try:
				editor_output = chat_text(
					messages=messages,
					model=EDITOR_MODEL,
					temperature=float(editor_temp),
				)
				st.session_state["editor_output"] = editor_output
				st.success("Готово: резюме сгенерировано")
			except Exception as e:
				st.error(f"Ошибка LLM: {e}")

if "editor_output" in st.session_state:
	st.subheader("Итог (Markdown с разделами)")
	st.markdown(st.session_state["editor_output"])  # Editor выводит Маркдаун и списки

st.divider()

