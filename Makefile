.PHONY: env env-dev lint test run

# Virtual environment creation and deps install.
env:
	conda create -n QISKitenv python=3 pip
	source activate QISKitenv
	pip install -r requires.txt

env-dev:
	pip install pylint

lint:
	pylint --rcfile=.rcfile qiskit test tools

test: lint
	python test

run:
	cd scripts
	jupyter notebook
