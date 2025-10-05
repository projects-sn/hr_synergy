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

ANALYZER_MODEL = os.getenv("ANALYZER_MODEL", "gpt-4o")
EDITOR_MODEL = os.getenv("EDITOR_MODEL", "gpt-4o")

st.set_page_config(page_title="Нейро‑HR — анализ и редактура резюме", layout="wide")
# Widen sidebar
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {width: 500px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎯 Нейро‑HR — анализ и редактура резюме")

with st.sidebar:
	st.header("Входные данные")
	resume_pdf = st.file_uploader("Загрузите PDF резюме", type=["pdf"])  # type: ignore
	job_description = st.text_area("Описание вакансии", height=450)


def load_resume_text() -> str:
    if resume_pdf is not None:
        # Save to temp and extract
        tmp_path = os.path.join(st.session_state.get("tmp_dir", "."), "_resume_tmp.pdf")
        with open(tmp_path, "wb") as f:
            f.write(resume_pdf.getbuffer())
        return extract_text_from_pdf(tmp_path)
    return ""


def _is_job_description_empty(text: str) -> bool:
    """Check JD emptiness per system rules: empty string, <20 words, or equals to N/A/none."""
    if not text or not text.strip():
        return True
    lowered = text.strip().lower()
    if lowered in {"n/a", "na", "none"}:
        return True
    # Count words conservatively
    words = [w for w in lowered.replace("\n", " ").split(" ") if w]
    return len(words) < 20


def _is_valid_analysis(analysis_json: dict) -> bool:
    """Minimal schema check for analyzer JSON to avoid UI errors."""
    if not isinstance(analysis_json, dict):
        return False
    required_top = [
        "overall_assessment",
        "clarity_assessment",
        "completeness_check",
        "volume_assessment",
        "keywords_match",
        "top_issues",
        "priority_fix_list",
    ]
    for key in required_top:
        if key not in analysis_json:
            return False
    return True


def format_analysis_report(analysis_json: dict) -> str:
	"""Форматирует JSON-отчёт Анализатора в человекочитаемый Markdown"""
	report = []
	
	# Общая оценка
	if "overall_assessment" in analysis_json:
		report.append(f"### Общая оценка\n{analysis_json['overall_assessment']}\n")
	
	# Оценка понятности
	if "clarity_assessment" in analysis_json:
		clarity = analysis_json["clarity_assessment"]
		rating = clarity.get("rating", "medium")
		report.append("### Оценка понятности")
		report.append(f"**Рейтинг:** {rating.upper()}")
		report.append(f"**Обоснование:** {clarity.get('why', 'Не указано')}")
		if clarity.get("suggestion"):
			report.append(f"**Рекомендация:** {clarity['suggestion']}")
		report.append("")
	
	# Оценка объема
	if "volume_assessment" in analysis_json:
		volume = analysis_json["volume_assessment"]
		report.append("### Оценка объема")
		# Отображаем как маркированный список, чтобы не сливалось в одну строку
		if volume.get("estimated_words"):
			report.append(f"- **Слов:** {volume['estimated_words']}")
		if volume.get("estimated_pages"):
			report.append(f"- **Страниц:** {volume['estimated_pages']}")
		if volume.get("relative_to_average"):
			report.append(f"- **Относительно среднего:** {volume['relative_to_average']}")
		if volume.get("relative_to_golden_standard"):
			report.append(f"- **Относительно эталона:** {volume['relative_to_golden_standard']}")
		if volume.get("why"):
			report.append(f"- **Почему:** {volume['why']}")
		if volume.get("suggestion"):
			report.append(f"- **Рекомендация:** {volume['suggestion']}")
		report.append("")
	
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
	if "completeness_check" in analysis_json and analysis_json["completeness_check"]:
		report.append("## Отсутствующие данные")
		for missing in analysis_json["completeness_check"]:
			field_name = missing.get('field', 'Неизвестное поле')
			status = missing.get('status', 'missing')
			field_display = {
				"contacts": "Контакты",
				"role": "Целевая должность",
				"seniority": "Уровень",
				"dates": "Даты работы", 
				"companies": "Компании",
				"responsibilities": "Обязанности",
				"achievements_metrics": "Достижения и метрики",
				"stack_tools": "Технологии и инструменты",
				"education": "Образование",
				"languages": "Языки",
				"location": "Местоположение",
				"links": "Ссылки"
			}.get(field_name, f"{field_name}")
			
			status_display = {
				"present": "✅",
				"partial": "⚠️",
				"missing": "❌"
			}.get(status, "❓")
			
			report.append(f"- **{field_display}:** {status_display} {missing.get('note', 'Не указано')}")
		report.append("")
	
	# Соответствие ключевых слов
	if "keywords_match" in analysis_json:
		keywords = analysis_json["keywords_match"]
		report.append("### Соответствие ключевых слов")
		
		# Показываем ключевые слова из JD (если есть)
		if keywords.get("from_jd"):
			report.append(f"**Ключевые слова из вакансии:** {', '.join(keywords['from_jd'])}")
		
		# Показываем найденные точные совпадения
		if keywords.get("found_exact"):
			report.append(f"**Найдено точно:** {', '.join(keywords['found_exact'])}")
		
		# Показываем найденные приблизительные совпадения
		if keywords.get("found_fuzzy"):
			report.append(f"**Найдено приблизительно:** {', '.join(keywords['found_fuzzy'])}")
		
		# Показываем отсутствующие ключевые слова
		if keywords.get("missing"):
			report.append(f"**Отсутствует:** {', '.join(keywords['missing'])}")
		
		# Показываем процент покрытия
		if keywords.get("coverage_percent") is not None:
			coverage = keywords['coverage_percent']
			coverage_emoji = "🟢" if coverage >= 80 else "🟡" if coverage >= 60 else "🔴"
			report.append(f"**Покрытие ключевых слов:** {coverage_emoji} {coverage}%")
		
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


st.header("🔹 Анализатор")
if st.button("Запустить анализ"):
    resume_text = load_resume_text()
    if not resume_text:
        st.warning("Требуется PDF резюме")
    else:
        # Normalize JD: if too short/missing, force empty to trigger generic analysis scenario
        jd_for_model = "" if _is_job_description_empty(job_description or "") else (job_description or "")
        user_prompt = ANALYZER_USER_TEMPLATE.format(
            resume_text=resume_text,
            job_description=jd_for_model,
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
                )
                
                # Проверяем на ошибки в ответе анализатора
                if isinstance(analysis_json, dict) and "error" in analysis_json:
                    reason = str(analysis_json.get("reason", "")).lower()
                    # Fallback: если модель ошиблась из-за пустого/короткого JD — принудительно повторим в режиме общего анализа
                    if any(k in reason for k in ["job description", "vacancy", "описание вакансии"]) and any(
                        k in reason for k in ["missing", "short", "корот", "нет", "пуст"]
                    ):
                        fallback_prompt = ANALYZER_USER_TEMPLATE.format(
                            resume_text=resume_text,
                            job_description="",
                        )
                        fallback_messages = [
                            {"role": "system", "content": ANALYZER_SYSTEM_PROMPT},
                            {"role": "user", "content": "ВНИМАНИЕ: описание вакансии отсутствует. Выполняй общий анализ, не возвращай ошибку."},
                            {"role": "user", "content": fallback_prompt},
                        ]
                        try:
                            analysis_json_fb = chat_json(
                                messages=fallback_messages,
                                model=ANALYZER_MODEL,
                            )
                            # Если вернулся объект-ошибка ИЛИ JSON не соответствует минимальным требованиям — попробуем строгую регенерацию
                            if (isinstance(analysis_json_fb, dict) and "error" in analysis_json_fb) or not _is_valid_analysis(analysis_json_fb):
                                strict_prompt = ANALYZER_USER_TEMPLATE.format(
                                    resume_text=resume_text,
                                    job_description="",
                                )
                                strict_messages = [
                                    {"role": "system", "content": ANALYZER_SYSTEM_PROMPT},
                                    {"role": "user", "content": "СТРОГИЙ РЕЖИМ: верни ТОЛЬКО валидный JSON строго по JSON Schema без ошибок и свободного текста."},
                                    {"role": "user", "content": strict_prompt},
                                ]
                                analysis_json_fb2 = chat_json(
                                    messages=strict_messages,
                                    model=ANALYZER_MODEL,
                                )
                                if isinstance(analysis_json_fb2, dict) and "error" not in analysis_json_fb2 and _is_valid_analysis(analysis_json_fb2):
                                    st.session_state["analysis_json"] = analysis_json_fb2
                                    st.info("Описание вакансии отсутствует или слишком короткое — выполнен общий анализ резюме.")
                                    st.success("Готово: отчёт сформирован")
                                    # Завершили обработку
                                    # (на уровне скрипта нельзя использовать return)
                                # Если снова не удалось — покажем исходную ошибку
                                st.error(f"Ошибка анализатора: {analysis_json.get('reason', 'Неизвестная ошибка')}")
                                # Завершение ветки строгой регенерации
                            if isinstance(analysis_json_fb, dict) and "error" not in analysis_json_fb:
                                st.session_state["analysis_json"] = analysis_json_fb
                                if _is_job_description_empty(job_description or ""):
                                    st.info("Описание вакансии отсутствует или слишком короткое — выполнен общий анализ резюме.")
                                else:
                                    st.info("Модель не смогла использовать описание вакансии — выполнен общий анализ резюме.")
                                st.success("Готово: отчёт сформирован")
                            else:
                                st.error(f"Ошибка анализатора: {analysis_json.get('reason', 'Неизвестная ошибка')}")
                        except Exception as _:
                            st.error(f"Ошибка анализатора: {analysis_json.get('reason', 'Неизвестная ошибка')}")
                    else:
                        st.error(f"Ошибка анализатора: {analysis_json.get('reason', 'Неизвестная ошибка')}")
                else:
                    # Если вернулся JSON без ключа error, но он не соответствует схеме — попросим строгую регенерацию
                    if not _is_valid_analysis(analysis_json):
                        strict_prompt = ANALYZER_USER_TEMPLATE.format(
                            resume_text=resume_text,
                            job_description=jd_for_model,
                        )
                        strict_messages = [
                            {"role": "system", "content": ANALYZER_SYSTEM_PROMPT},
                            {"role": "user", "content": "СТРОГИЙ РЕЖИМ: верни ТОЛЬКО валидный JSON строго по JSON Schema без ошибок и свободного текста."},
                            {"role": "user", "content": strict_prompt},
                        ]
                        try:
                            analysis_json2 = chat_json(
                                messages=strict_messages,
                                model=ANALYZER_MODEL,
                            )
                            if isinstance(analysis_json2, dict) and "error" not in analysis_json2 and _is_valid_analysis(analysis_json2):
                                st.session_state["analysis_json"] = analysis_json2
                                st.success("Готово: отчёт сформирован")
                            else:
                                st.error("Ошибка анализатора: ответ не соответствует JSON Schema")
                        except Exception as _:
                            st.error("Ошибка анализатора: ответ не соответствует JSON Schema")
                    else:
                        st.session_state["analysis_json"] = analysis_json
                        st.success("Готово: отчёт сформирован")
            except Exception as e:
                st.error(f"Ошибка LLM: {e}")

# Показываем результаты анализа (только форматированный отчёт)
if "analysis_json" in st.session_state:
    analysis_json = st.session_state["analysis_json"]
    st.markdown(format_analysis_report(analysis_json))


st.header("🔹 Редактор")
# Version selector for editor stage (Russian labels)
col_v1, col_v2 = st.columns([1, 3])
with col_v1:
    resume_version_label = st.radio("Версия резюме", options=["Короткая", "Расширенная"], index=0)
resume_version = "concise" if resume_version_label == "Короткая" else "full"

if st.button("Сгенерировать улучшенное резюме"):
    resume_text = load_resume_text()
    if not resume_text:
        st.warning("Требуется PDF резюме")
    else:
        if "analysis_json" not in st.session_state:
            st.info("Сначала запустите Анализатор — его вывод используется Редактором")
        analysis_json_str = orjson.dumps(st.session_state.get("analysis_json", {})).decode()
        user_prompt = EDITOR_USER_TEMPLATE.format(
            analyzer_json=analysis_json_str,
            resume_text=resume_text,
            job_description=job_description or "",
            resume_version=resume_version,
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
                )
                
                # Проверяем на ошибки в ответе редактора
                if editor_output.strip().startswith('{"error":'):
                    try:
                        error_json = orjson.loads(editor_output)
                        st.error(f"Ошибка редактора: {error_json.get('details', 'Неизвестная ошибка')}")
                    except:
                        st.error(f"Ошибка редактора: {editor_output}")
                else:
                    st.session_state["editor_output"] = editor_output
                    st.success("Готово: резюме сгенерировано")
            except Exception as e:
                st.error(f"Ошибка LLM: {e}")

if "editor_output" in st.session_state:
	st.subheader("Итог (Markdown с разделами)")
	st.markdown(st.session_state["editor_output"])  # Editor выводит Маркдаун и списки

st.divider()

st.header("🔹 Оценка зарплатной вилки (RUB/мес)")
with st.expander("Показать/скрыть оценку зарплаты"):
	try:
		from salary_estimator import estimate_salary_from_resume
		_resume_text = load_resume_text()
		if not _resume_text:
			st.info("Загрузите PDF резюме, чтобы автоматически оценить вилку и подходящие роли.")
		else:
			result = estimate_salary_from_resume(
				resume_text=_resume_text,
				job_description=job_description or None,
				model=ANALYZER_MODEL,
				temperature=0.1,
			)
			st.subheader("Подходящие направления и профессии")
			roles = result.get("roles", []) or []
			if roles:
				for r in roles:
					title = r.get("title", "—")
					dirn = r.get("direction", "—")
					sen = r.get("seniority") or "—"
					reason = r.get("fit_reason", "")
					st.markdown(f"- **{title}** — {dirn} ({sen})  ")
					if reason:
						st.caption(reason)
			st.subheader("Оценка рыночной вилки (RUB/мес)")
			est_overall = result.get("estimate_rub_month", {})
			if est_overall:
				st.markdown(f"**Итого:** {est_overall.get('min', '—')} — {est_overall.get('max', '—')} (медиана: {est_overall.get('median', '—')})")
			ranges = result.get("ranges_per_role", []) or []
			if ranges:
				with st.expander("Диапазоны по ролям"):
					for rr in ranges:
						st.write(f"- {rr.get('title', '—')}: {rr.get('min', '—')} — {rr.get('max', '—')} (медиана: {rr.get('median', '—')})")
			conf = result.get("confidence", "—")
			st.markdown(f"**Доверие:** {conf}")
			notes = result.get("notes", "")
			if notes:
				st.markdown(f"**Примечания:** {notes}")
			sources = result.get("sources", [])
			if sources:
				with st.expander("Источники"):
					for s in sources:
						st.write(f"- {s}")
			assumptions = result.get("assumptions", [])
			if assumptions:
				with st.expander("Допущения"):
					for a in assumptions:
						st.write(f"- {a}")
	except Exception as e:
		st.error(f"Ошибка при оценке зарплаты: {e}")
