.PHONY: test lint type fmt proto bench build all

UV ?= uv

all: test lint type

test:
	$(UV) run pytest --cov --cov-report=term

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

fmt:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

type:
	$(UV) run mypy

proto:
	$(UV) run python -m grpc_tools.protoc -I src/senseai_ingest/schema \
		--python_out=src/senseai_ingest/schema --pyi_out=src/senseai_ingest/schema \
		src/senseai_ingest/schema/sample.proto

bench:
	$(UV) run python benchmarks/bench_aligner.py

build:
	$(UV) build
