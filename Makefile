.PHONY: help setup dev health-check smoke test test-unit lint build clean stop install-tools

help:
	@echo Available commands:
	@echo   dev          - Start local development environment
	@echo   health-check - Check service health
	@echo   smoke        - Run smoke tests against localhost:8000
	@echo   test         - Run full test suite
	@echo   test-unit    - Run unit tests only (skip integration)
	@echo   lint         - Auto-fix lint and formatting
	@echo   clean        - Clean up resources

dev:
	@echo Starting local development...
	docker-compose up --build -d
	@echo Services started at:
	@echo   API: http://localhost:8000
	@echo   MLflow: http://localhost:5000
	@echo   Grafana: http://localhost:3000

health-check:
	@echo Checking service health...
	docker-compose ps
	python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health').read().decode())"

smoke:
	python scripts/smoke_test.py

test:
	poetry run pytest tests/ -v

test-unit:
	poetry run pytest tests/ -v -m "not integration"

lint:
	poetry run ruff check app ml tests scripts --fix
	poetry run black app ml tests scripts

build:
	docker build -t diabetes-mlops:dev .

clean:
	docker-compose down -v
	docker system prune -f

stop:
	docker-compose down

install-tools:
	poetry run pip install --upgrade pre-commit ruff black
