.PHONY: generate-data train build-index run-api run-dashboard run-all test lint

generate-data:
	python data/generate_data.py

train:
	python ml/train.py

build-index:
	python rag/build_index.py

run-api:
	uvicorn api.main:app --reload --port 8000

run-dashboard:
	streamlit run dashboard/app.py

run-all:
	docker-compose up --build

test:
	pytest tests/ -v

lint:
	ruff check .
