lint:
	flake8 src

format:
	black src
	isort src

tests:
	python -B src/tests/main.py

prepare:
	python -B src/prepare.py

train:
	python src/train.py

test:
	python src/test.py