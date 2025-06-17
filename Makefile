.PHONY: help setup dev health-check test lint build clean install-tools

help:
	@echo Available commands:
	@echo   dev          - Start local development environment
	@echo   health-check - Check service health
	@echo   test         - Run tests
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
	@powershell -Command "try { Invoke-RestMethod http://localhost:8000/health } catch { Write-Host 'API not ready yet' }"

test:
	poetry run pytest tests/ -v

lint:
	poetry run ruff check . --fix
	poetry run black .

build:
	docker build -t diabetes-mlops:dev .

clean:
	docker-compose down -v
	docker system prune -f

stop:
	docker-compose down

install-tools:
	poetry run pip install --upgrade pre-commit ruff black
