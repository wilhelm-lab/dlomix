install:
	pip install --upgrade pip && pip install -e .

install-dev:
	pip install --upgrade pip && pip install -e .[dev]

.PHONY: test
test:
	python -m pytest ./test/*.py --doctest-modules --junitxml=junit/test-results.xml --cov=dlomix --cov-report=xml --cov-report=html

format:
	black ./dlomix/*

lint:
	pylint --disable=R,C ./dlomix/*

all: install format test
