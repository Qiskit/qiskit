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
	pylint qiskit test

# TODO: Uncomment when the lint one passes.
# test: lint
test:
	python3 -m unittest discover -v

profile:
	python3 -m unittest discover -p "profile*.py" -v

doc:
	export PYTHONPATH=$(PWD); \
	better-apidoc -f -o doc/_autodoc -d 5 -e -t doc/_templates/better-apidoc qiskit qiskit/tools "qiskit/extensions/standard/[a-z]*"; \
	sphinx-autogen -t doc/_templates doc/_autodoc/*; \
	make -C doc html

clean:
	make -C doc clean
