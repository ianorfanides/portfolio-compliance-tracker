install:
	uv venv
	.venv/Scripts/pip install fastapi uvicorn "google-adk>=2.0.0a0" "pydantic>=2.0.0" "python-dotenv>=1.0.0"

playground:
	.venv/Scripts/adk web

serve:
	.venv/Scripts/uvicorn src.app:app --host 0.0.0.0 --port 8080
