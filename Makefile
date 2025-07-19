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

# Documentation

BACKEND ?= tensorflow
DOCS_DIR := docs
BUILD_DIR := $(DOCS_DIR)/_build/html/$(BACKEND)

build-docs-framework:
	sphinx-apidoc -M -f -E -l -o $(DOCS_DIR)/ src/dlomix/
	python $(DOCS_DIR)/codify_package_titles.py
	DLOMIX_BACKEND=$(BACKEND) sphinx-build -b html $(DOCS_DIR)/ $(BUILD_DIR)
	open $(BUILD_DIR)/index.html

build-docs:
	$(MAKE) build-docs-framework BACKEND=tensorflow
	$(MAKE) build-docs-framework BACKEND=pytorch


all: install format test
