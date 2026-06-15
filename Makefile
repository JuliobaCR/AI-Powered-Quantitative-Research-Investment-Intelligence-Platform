.PHONY: run test lint install clean

install:
	pip install -r requirements.txt

run:
	streamlit run src/dashboard/app.py

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/
	black --check src/ tests/

format:
	black src/ tests/
	ruff check --fix src/ tests/

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
