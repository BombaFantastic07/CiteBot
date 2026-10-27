.PHONY: install ingest chat serve test lint eval-retrieval eval-hallucination bench docker

install:
	pip install -r requirements.txt

ingest:
	ragbot ingest

chat:
	ragbot chat

serve:
	uvicorn ragbot.api:app --host 0.0.0.0 --port 8000

test:
	pytest -q

lint:
	ruff check src tests evaluation

eval-retrieval:
	python evaluation/eval_retrieval.py --dataset evaluation/dataset.jsonl

eval-hallucination:
	python evaluation/eval_hallucination.py --dataset evaluation/dataset.jsonl

bench:
	python evaluation/benchmark_latency.py --dataset evaluation/dataset.jsonl --runs 3

docker:
	docker build -t ragbot .
