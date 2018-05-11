# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
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
	for LANGUAGE in "." "ja"; do \
		better-apidoc -f -o doc/$$LANGUAGE/_autodoc -d 5 -e -t doc/_templates/better-apidoc qiskit qiskit/tools "qiskit/extensions/standard/[a-z]*"; \
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
	make -C doc -e BUILDDIR="_build/ja" -e SOURCEDIR="./ja" clean
	make -C src/qasm-simulator-cpp/src clean
	rm -f test/python/test_latex_drawer.tex test/python/test_qasm_python_simulator.pdf \
		test/python/test_save.json test/python/test_teleport.tex
