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

build-docs:
	cd docs && make clean html
	cd docs/_build/html/ && open index.html

all: install format test
