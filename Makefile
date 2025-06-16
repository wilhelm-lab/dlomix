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
	mkdir -p .test_results/cov/

	python -m pytest tests/ --junitxml=.test_results/junit/test-results.xml --cov=dlomix --cov-report html:.test_results/cov/cov_html --cov-report xml:.test_results/cov/cov.xml --cov-report lcov:.test_results/cov/cov.info --cov-report annotate:.test_results/cov/cov_annotate

test-local:
	make uninstall
	make install-nodeps
	mkdir -p .test_results/cov/

	python -m pytest tests/ --junitxml=.test_results/junit/test-results.xml --cov=dlomix --cov-report html:.test_results/cov/cov_html --cov-report xml:.test_results/cov/cov.xml --cov-report lcov:.test_results/cov/cov.info --cov-report annotate:.test_results/cov/cov_annotate


format:
	black ./src/dlomix/*
	isort --profile black .
	black ./src/dlomix/*.py
	black ./run_scripts/*.py
	black ./tests/*.py

lint:
	pylint --disable=R,C ./src/dlomix/*

lint-errors-only:
	pylint --errors-only --disable=R,C ./src/dlomix/*

build-docs:
	sphinx-apidoc -M -f -E -l -o docs/ src/dlomix/
	python docs/codify_package_titles.py
	cd docs && make clean html
	cd docs/_build/html/ && open index.html

all: install format test
