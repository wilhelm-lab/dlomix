install:
	pip install --upgrade pip && pip install -e .

uninstall:
	pip uninstall dlomix -y

install-nodeps:
	pip install --upgrade pip && pip install -e . --no-deps


install-dev:
	pip install --upgrade pip && pip install -e .[dev]

.PHONY: test
test:
	make uninstall
	make install
	mkdir -p cov/

	python -m pytest tests/ --junitxml=junit/test-results.xml --cov=dlomix --cov-report html:cov/cov_html --cov-report xml:cov/cov.xml --cov-report lcov:cov/cov.info --cov-report annotate:cov/cov_annotate

test-local:
	make uninstall
	make install-nodeps
	mkdir -p cov/

	python -m pytest tests/ --junitxml=junit/test-results.xml --cov=dlomix --cov-report html:cov/cov_html --cov-report xml:cov/cov.xml --cov-report lcov:cov/cov.info --cov-report annotate:cov/cov_annotate


format:
	black ./dlomix/*
	isort --profile black .
	black ./dlomix/*.py
	black ./run_scripts/*.py
	black ./tests/*.py

lint:
	pylint --disable=R,C ./dlomix/*

build-docs:
	cd docs && make clean html
	cd docs/_build/html/ && open index.html

all: install format test
