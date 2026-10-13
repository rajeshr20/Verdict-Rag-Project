.PHONY: install run test eval docker-up ingest-samples clean

# Python virtual environment binary detection
ifeq ($(OS),Windows_NT)
    PYTHON = .venv\Scripts\python
    PIP = .venv\Scripts\pip
    PYTEST = .venv\Scripts\pytest
else
    PYTHON = .venv/bin/python
    PIP = .venv/bin/pip
    PYTEST = .venv/bin/pytest
endif

install:
	$(PIP) install -r requirements.txt

run:
	$(PYTHON) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

streamlit:
	$(PYTHON) -m streamlit run streamlit_app.py --server.port 8501

test:
	$(PYTEST) -v --cov=app/core --cov-report=term-missing

eval:
	$(PYTHON) eval/run_eval.py

docker-up:
	docker compose up --build -d

ingest-samples:
	$(PYTHON) scripts/ingest_samples.py

list-models:
	$(PYTHON) -m scripts.list_models

clean:
	rm -rf .pytest_cache .coverage htmlcov eval_chroma_db
