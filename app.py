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
from salary_estimator import estimate_salary_from_resume

ANALYZER_MODEL = os.getenv("ANALYZER_MODEL", "gpt-4o-mini")
EDITOR_MODEL = os.getenv("EDITOR_MODEL", "gpt-4o")

st.set_page_config(page_title="Нейро‑HR — анализ и редактура резюме", layout="wide")

st.title("🎯 Нейро‑HR — анализ и редактура резюме")

with st.sidebar:
	st.header("Входные данные")
	resume_pdf = st.file_uploader("Загрузите PDF резюме", type=["pdf"])  # type: ignore
	job_description = st.text_area("Описание вакансии", height=180)


def load_resume_text() -> str:
	if resume_pdf is not None:
		# Save to temp and extract
		tmp_path = os.path.join(st.session_state.get("tmp_dir", "."), "_resume_tmp.pdf")
		with open(tmp_path, "wb") as f:
			f.write(resume_pdf.getbuffer())
		return extract_text_from_pdf(tmp_path)
	return ""


def format_analysis_report(analysis_json: dict) -> str:
	"""Форматирует JSON-отчёт Анализатора в человекочитаемый Markdown"""
	report = []
	
	# Список обработанных полей, чтобы потом вывести остальные
	processed_fields = set()
	
	# Общая оценка
	if "overall_assessment" in analysis_json:
		report.append(f"### Общая оценка\n{analysis_json['overall_assessment']}\n")
		processed_fields.add("overall_assessment")
	
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
		processed_fields.add("top_issues")
	
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
		processed_fields.add("missing_data")
	
	# Соответствие ключевых слов
	if "keywords_match" in analysis_json:
		keywords = analysis_json["keywords_match"]
		report.append("### Соответствие ключевых слов")
		if keywords.get("found_in_resume"):
			report.append(f"**Найдено в резюме:** {', '.join(keywords['found_in_resume'])}")
		if keywords.get("missing"):
			report.append(f"**Отсутствует:** {', '.join(keywords['missing'])}")
		report.append("")
		processed_fields.add("keywords_match")
	
	# Риски
	if "risks" in analysis_json and analysis_json["risks"]:
		report.append("### Риски форматирования")
		for risk in analysis_json["risks"]:
			report.append(f"- {risk}")
		report.append("")
		processed_fields.add("risks")
	
	# Вопросы кандидату
	if "candidate_questions" in analysis_json and analysis_json["candidate_questions"]:
		report.append("### Вопросы для уточнения")
		for i, question in enumerate(analysis_json["candidate_questions"], 1):
			report.append(f"{i}. {question}")
		report.append("")
		processed_fields.add("candidate_questions")
	
	# Приоритетный список исправлений
	if "priority_fix_list" in analysis_json and analysis_json["priority_fix_list"]:
		report.append("### План действий")
		for i, fix in enumerate(analysis_json["priority_fix_list"], 1):
			report.append(f"**{i}.** {fix}")
		report.append("")
		processed_fields.add("priority_fix_list")
	
	# Обработка всех остальных полей (для дополнительных оценок и метрик)
	for key, value in analysis_json.items():
		if key in processed_fields:
			continue
		
		# Пропускаем пустые значения (None, пустые строки, пустые списки/словари)
		if value is None:
			continue
		if isinstance(value, str) and not value.strip():
			continue
		if isinstance(value, (list, dict)) and len(value) == 0:
			continue
		
		# Форматируем название поля (заглавная буква, замена подчеркиваний)
		# Специальная обработка для полей "оценка_*"
		if key.startswith("оценка_") or "оценка" in key.lower():
			field_title = key.replace("_", " ").title()
		else:
			field_title = key.replace("_", " ").title()
		
		# Обработка разных типов значений
		if isinstance(value, str):
			report.append(f"### {field_title}\n{value}\n")
		elif isinstance(value, dict):
			report.append(f"### {field_title}")
			for sub_key, sub_value in value.items():
				sub_title = str(sub_key).replace("_", " ").title()
				# Форматируем название поля на русском
				if sub_key == "рейтинг":
					sub_title = "Рейтинг"
				elif sub_key == "обоснование":
					sub_title = "Обоснование"
				elif sub_key == "статус":
					sub_title = "Статус"
				elif sub_key == "justification":
					sub_title = "Обоснование"
				elif sub_key == "reason":
					sub_title = "Обоснование"
				elif sub_key == "rating":
					sub_title = "Рейтинг"
				
				if isinstance(sub_value, str):
					report.append(f"**{sub_title}:** {sub_value}")
				elif isinstance(sub_value, (list, dict)) and len(sub_value) > 0:
					report.append(f"**{sub_title}:** {sub_value}")
				else:
					report.append(f"**{sub_title}:** {sub_value}")
			report.append("")
		elif isinstance(value, list):
			report.append(f"### {field_title}")
			for item in value:
				if isinstance(item, str):
					report.append(f"- {item}")
				elif isinstance(item, dict):
					# Если это объект, выводим его поля
					for item_key, item_value in item.items():
						item_title = str(item_key).replace("_", " ").title()
						report.append(f"  - **{item_title}:** {item_value}")
				else:
					report.append(f"- {item}")
			report.append("")
		else:
			report.append(f"### {field_title}\n{value}\n")
	
	return "\n".join(report)


def format_salary_report(salary_json: dict) -> str:
	"""Форматирует JSON-отчёт оценки зарплаты в человекочитаемый Markdown"""
	report = []
	
	# Общая оценка зарплаты
	if "estimate_rub_month" in salary_json:
		est = salary_json["estimate_rub_month"]
		report.append("### 💰 Оценка зарплаты")
		min_val = est.get('min', 0)
		max_val = est.get('max', 0)
		if min_val and max_val:
			report.append(f"**Диапазон:** {min_val:,} — {max_val:,} руб/мес")
		else:
			report.append(f"**Диапазон:** не указано")
		if "median" in est and est.get('median'):
			report.append(f"**Медиана:** {est['median']:,} руб/мес")
		report.append("")
	
	# Роли
	if "roles" in salary_json and salary_json["roles"]:
		report.append("### Подходящие роли")
		for i, role in enumerate(salary_json["roles"], 1):
			report.append(f"#### {i}. {role.get('title', 'Не указано')}")
			if role.get('direction'):
				report.append(f"**Направление:** {role['direction']}")
			if role.get('seniority'):
				report.append(f"**Уровень:** {role['seniority']}")
			if role.get('fit_reason'):
				report.append(f"**Почему подходит:** {role['fit_reason']}")
			report.append("")
	
	# Диапазоны по ролям
	if "ranges_per_role" in salary_json and salary_json["ranges_per_role"]:
		report.append("### Диапазоны зарплат по ролям")
		for role_range in salary_json["ranges_per_role"]:
			title = role_range.get('title', 'Не указано')
			min_sal = role_range.get('min', 0)
			max_sal = role_range.get('max', 0)
			median_sal = role_range.get('median', 0)
			report.append(f"**{title}:** {min_sal:,} — {max_sal:,} руб/мес (медиана: {median_sal:,})")
		report.append("")
	
	# Уверенность
	if "confidence" in salary_json:
		confidence = salary_json["confidence"]
		confidence_ru = {
			"high": "высокая",
			"medium": "средняя",
			"low": "низкая"
		}.get(confidence, confidence)
		report.append(f"**Уверенность оценки:** {confidence_ru}")
		report.append("")
	
	# Допущения
	if "assumptions" in salary_json and salary_json["assumptions"]:
		report.append("### Допущения")
		for assumption in salary_json["assumptions"]:
			report.append(f"- {assumption}")
		report.append("")
	
	# Источники
	if "sources" in salary_json and salary_json["sources"]:
		report.append("### Источники")
		for source in salary_json["sources"]:
			report.append(f"- {source}")
		report.append("")
	
	# Примечания
	if "notes" in salary_json and salary_json["notes"]:
		report.append("### Примечания")
		report.append(salary_json["notes"])
		report.append("")
	
	return "\n".join(report)


st.header("🔹 Анализатор")
if st.button("Запустить анализ"):
	resume_text = load_resume_text()
	if not resume_text:
		st.warning("Требуется загрузить PDF резюме")
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
					temperature=0.1,
				)
				st.session_state["analysis_json"] = analysis_json
				st.success("Готово: отчёт сформирован")
			except Exception as e:
				st.error(f"Ошибка LLM: {e}")

# Показываем результаты анализа
if "analysis_json" in st.session_state:
	analysis_json = st.session_state["analysis_json"]
	st.markdown(format_analysis_report(analysis_json))


st.header("🔹 Редактор")
if st.button("Сгенерировать улучшенное резюме"):
	resume_text = load_resume_text()
	if not resume_text:
		st.warning("Требуется загрузить PDF резюме")
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
					temperature=0.3,
				)
				st.session_state["editor_output"] = editor_output
				st.success("Готово: резюме сгенерировано")
			except Exception as e:
				st.error(f"Ошибка LLM: {e}")

if "editor_output" in st.session_state:
	st.subheader("Итог (Markdown с разделами)")
	st.markdown(st.session_state["editor_output"])  # Editor выводит Маркдаун и списки


st.header("🔹 Оценка зарплаты")
if st.button("Оценить зарплату"):
	resume_text = load_resume_text()
	if not resume_text:
		st.warning("Требуется загрузить PDF резюме")
	else:
		with st.spinner("Модель оценивает зарплату…"):
			try:
				salary_json = estimate_salary_from_resume(
					resume_text=resume_text,
					job_description=job_description or None,
				)
				st.session_state["salary_json"] = salary_json
				st.success("Готово: оценка зарплаты сформирована")
			except Exception as e:
				st.error(f"Ошибка LLM: {e}")

# Показываем результаты оценки зарплаты
if "salary_json" in st.session_state:
	salary_json = st.session_state["salary_json"]
	st.markdown(format_salary_report(salary_json))

st.divider()

