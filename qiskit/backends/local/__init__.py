# -*- coding: utf-8 -*-

# Copyright 2018 IBM RESEARCH. All Rights Reserved.
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

"""Local Backends."""

from .localprovider import LocalProvider
from .localjob import LocalJob
from .qasm_simulator_cpp import CliffordSimulatorCpp, QasmSimulatorCpp
from .qasm_simulator_py import QasmSimulatorPy
from .qasm_simulator_projectq import QasmSimulatorProjectQ
from .statevector_simulator_cpp import StatevectorSimulatorCpp
from .statevector_simulator_py import StatevectorSimulatorPy
from .statevector_simulator_sympy import StatevectorSimulatorSympy
from .unitary_simulator_py import UnitarySimulatorPy
from .unitary_simulator_sympy import UnitarySimulatorSympy
