.PHONY: lint format
lint:
	python -m ruff check src scripts

format:
	python -m ruff format src scripts
