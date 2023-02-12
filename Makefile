install:
	pip install --upgrade pip && pip install -e .

install-dev:
	pip install --upgrade pip && pip install -e .[dev]

.PHONY: test
test:
	mkdir -p cov/

	python -m pytest tests/ --junitxml=junit/test-results.xml --cov=dlomix --cov-report html:cov/cov_html --cov-report xml:cov/cov.xml --cov-report lcov:cov/cov.info --cov-report annotate:cov/cov_annotate

format:
	isort .
	black ./dlomix/*.py
	black ./run_scripts/*.py
	black ./tests/*.py
	
lint:
	pylint --disable=R,C ./dlomix/*

all: install format test
