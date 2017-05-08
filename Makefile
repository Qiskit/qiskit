.PHONY: env env-dev lint test run

# Virtual environment creation and deps install.
env:
	conda create -n QISKitenv python=3 pip
	source activate QISKitenv
	pip install -r requires.txt

env-dev:
	pip install pylint

lint:
	pylint qiskit test tools

test: lint
#	py.test --verbose --color=yes $(TEST_PATH)
	python test

run:
	jupyter notebook
