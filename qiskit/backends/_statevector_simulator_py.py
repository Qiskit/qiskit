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

The input is a qobj dictionary

and the output is a Result object

    result['data']['quantum_state']

where 'quantum_state' is a 2 :sup:`n` complex numpy array representing the
quantum state vector

The simulator is run using

.. code-block:: python

    StatevectorSimulatorPy(compiled_circuit,shots,seed).run().

.. code-block:: guess

       compiled_circuit =
       {
        "header": {
        "number_of_qubits": 2, // int
        "number_of_clbits": 2, // int
        "qubit_labels": [["q", 0], ["v", 0]], // list[list[string, int]]
        "clbit_labels": [["c", 2]], // list[list[string, int]]
        }
        "operations": // list[map]
           [
               {
                   "name": , // required -- string
                   "params": , // optional -- list[double]
                   "qubits": , // required -- list[int]
                   "clbits": , // optional -- list[int]
                   "conditional":  // optional -- map
                       {
                           "type": , // string
                           "mask": , // hex string
                           "val":  , // bhex string
                       }
               },
           ]
       }

.. code-block:: python

       result =
               {
               'data':
                   {
                   'quantum_state': array([ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j]),
                   'classical_state': 0
                   'counts': {'0000': 1}
                   }
               'status': 'DONE'
               }
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
from qiskit.backends._qasm_simulator_py import QasmSimulatorPy

logger = logging.getLogger(__name__)

# TODO add ["status"] = 'DONE', 'ERROR' especitally for empty circuit error
# does not show up

class StatevectorSimulatorPy(QasmSimulatorPy):
    """Python implementation of a statevector simulator."""

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
                'description': 'A statevector simulator implemented in Python',
                'coupling_map': 'all-to-all',
                'basis_gates': 'u1,u2,u3,cx,id,snapshot'
            }
        else:
            self._configuration = configuration

        self._local_random = random.Random()

        # Define attributes in __init__.
        self._classical_state = 0
        self._quantum_state = 0
        self._snapshots = {}
        self._number_of_cbits = 0
        self._number_of_qubits = 0
        self._shots = 0

    def validate(self, qobj):
        return
