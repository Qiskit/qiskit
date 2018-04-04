# -*- coding: utf-8 -*-
# pylint: disable=invalid-name

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

"""Contains a (slow) python statevector simulator.

It simulates the statevector through a quantum circuit. It is exponential in
the number of qubits.

We advise using the c++ simulator or online simulator for larger size systems.

The input is a qobj dictionary and the output is a Result object.

The input qobj to this simulator has no shots, no measures, no reset, no noise.
"""
import random
import uuid
import logging
from collections import Counter

import numpy as np

from qiskit._result import Result
from qiskit.backends._basebackend import BaseBackend
from ._simulatorerror import SimulatorError
from ._simulatortools import single_gate_matrix
from ._qasm_simulator_py import QasmSimulatorPy

logger = logging.getLogger(__name__)


class StatevectorSimulatorPy(QasmSimulatorPy):
    """Python statevector simulator."""

    def __init__(self, configuration=None):
        """
        Args:
            configuration (dict): backend configuration
        """
        super().__init__(configuration)
        if configuration is None:
            self._configuration = {
                'name': 'local_statevector_simulator_py',
                'url': 'https://github.com/QISKit/qiskit-sdk-py',
                'simulator': True,
                'local': True,
                'description': 'A python statevector simulator for qobj files',
                'coupling_map': 'all-to-all',
                'basis_gates': 'u1,u2,u3,cx,id,snapshot'
            }
        else:
            self._configuration = configuration

    def run(self, q_job):
        """Run circuits in q_job."""
        return super().run(q_job)

    def validate(self, qobj):
        return True
