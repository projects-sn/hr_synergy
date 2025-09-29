from __future__ import annotations

from typing import Iterable

from pypdf import PdfReader


def extract_text_from_pdf(file_path: str) -> str:
	reader = PdfReader(file_path)
	texts: list[str] = []
	for page in reader.pages:
		text = page.extract_text() or ""
		texts.append(text)
	raw = "\n\n".join(texts)
	return normalize_whitespace(raw)


def normalize_whitespace(text: str) -> str:
	lines = [" ".join(line.split()) for line in text.splitlines()]
	return "\n".join(line for line in lines if line is not None)
