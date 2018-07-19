# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

.PHONY: env env-dev lint test run doc

# Dependencies need to be installed on the Anaconda virtual environment.
env:
	if test $(findstring QISKitenv, $(shell conda info --envs)); then \
		bash -c "source activate QISKitenv;pip install -r requirements.txt"; \
	else \
		conda create -y -n QISKitenv python=3; \
		bash -c "source activate QISKitenv;pip install -r requirements.txt"; \
	fi;

run:
	bash -c "source activate QISKitenv;cd examples; cd jupyter;jupyter notebook"

# Ignoring generated ones with .py extension.
lint:
	pylint -rn qiskit test

style:
	pycodestyle --max-line-length=100 qiskit test

# Use the -s (starting directory) flag for "unittest discover" is necessary,
# otherwise the QuantumCircuit header will be modified during the discovery.
test:
	python3 -m unittest discover -s test -v

profile:
	python3 -m unittest discover -p "profile*.py" -v

coverage:
	coverage3 run --source qiskit -m unittest discover -s test -q
	coverage3 report

doc:
	export PYTHONPATH=$(PWD); \
	for LANGUAGE in "." "de" "ja"; do \
		better-apidoc -f -o doc/$$LANGUAGE/_autodoc --no-toc --private --maxdepth=5 --separate --templates=doc/_templates/better-apidoc qiskit qiskit/tools "qiskit/extensions/standard/[a-z]*"; \
		sphinx-autogen -t doc/_templates doc/$$LANGUAGE/_autodoc/*; \
		make -C doc -e BUILDDIR="_build/$$LANGUAGE" -e SOURCEDIR="./$$LANGUAGE" html; \
	done

sim:
	make -C src/qasm-simulator-cpp/src clean
	make -C src/qasm-simulator-cpp/src

# Build dependencies for the simulator target
depend:
	make -C src/qasm-simulator-cpp depend

coverage_erase:
	coverage erase

clean: coverage_erase
	make -C doc clean
	make -C doc -e BUILDDIR="_build/de" -e SOURCEDIR="./de" clean
	make -C doc -e BUILDDIR="_build/ja" -e SOURCEDIR="./ja" clean
	make -C src/qasm-simulator-cpp/src clean
	rm -f test/python/test_latex_drawer.tex test/python/test_qasm_python_simulator.pdf \
		test/python/test_save.json test/python/test_teleport.tex
