.PHONY: install train serve test lint format docker-build docker-up docker-down clean help

PYTHON  := python
SRC_DIR := src
DATA    := data/raw/CIC-DDoS2019.csv

help:
	@echo ""
	@echo "DDoS Attack Detection — available commands:"
	@echo ""
	@echo "  make install        Install Python dependencies"
	@echo "  make train          Train all models (set DATA= to override path)"
	@echo "  make train-merge    Train using all CSVs in data/raw/"
	@echo "  make serve          Start FastAPI server on :8000"
	@echo "  make test           Run pytest with coverage"
	@echo "  make lint           Run flake8 + mypy"
	@echo "  make format         Auto-format with black + isort"
	@echo "  make docker-build   Build Docker image"
	@echo "  make docker-up      Start API via docker-compose"
	@echo "  make docker-down    Stop docker-compose services"
	@echo "  make clean          Remove compiled Python files"
	@echo ""

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

train:
	PYTHONPATH=$(SRC_DIR) $(PYTHON) $(SRC_DIR)/train.py --data $(DATA)

train-merge:
	PYTHONPATH=$(SRC_DIR) $(PYTHON) $(SRC_DIR)/train.py --data data/raw/ --merge

serve:
	PYTHONPATH=$(SRC_DIR) uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

test:
	PYTHONPATH=$(SRC_DIR) pytest tests/ -v --cov=$(SRC_DIR) --cov-report=term-missing

lint:
	flake8 $(SRC_DIR)/ tests/ --max-line-length=120 --extend-ignore=E203,W503
	mypy $(SRC_DIR)/ --ignore-missing-imports

format:
	black $(SRC_DIR)/ tests/
	isort $(SRC_DIR)/ tests/

docker-build:
	docker build -t ddos-attack-detection:latest .

docker-up:
	docker-compose up --build -d api

docker-down:
	docker-compose down

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
