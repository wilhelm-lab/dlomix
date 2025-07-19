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
BUILD_ROOT := $(DOCS_DIR)/_build
BUILD_DIR := $(BUILD_ROOT)/html/$(BACKEND)

build-docs-framework:
	sphinx-apidoc -M -f -E -l -o $(DOCS_DIR)/ src/dlomix/
	python $(DOCS_DIR)/codify_package_titles.py
	DLOMIX_BACKEND=$(BACKEND) sphinx-build -b html $(DOCS_DIR)/ $(BUILD_DIR)

build-docs:
	# Clean old builds
	rm -rf $(BUILD_ROOT)/html

	# Build TensorFlow
	$(MAKE) build-docs-framework BACKEND=tensorflow

	# Build PyTorch
	$(MAKE) build-docs-framework BACKEND=pytorch

	$(MAKE) create-root-index

	# Optional: open main page
	open $(BUILD_ROOT)/html/index.html

create-root-index:
	echo '<html><head><meta http-equiv="refresh" content="0; url=tensorflow/index.html"></head></html>' > $(BUILD_ROOT)/html/index.html


all: install format test
