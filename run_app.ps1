$ErrorActionPreference = "Stop"

# Ensure venv usage if present
if (Test-Path .venv) {
	. .\.venv\Scripts\Activate.ps1
}

$env:PYTHONIOENCODING = "utf-8"
streamlit run app.py --server.headless true

