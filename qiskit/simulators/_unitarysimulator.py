# -*- coding: utf-8 -*-

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

"""Contains a (slow) Python simulator that returns the unitary of the circuit.

Author: Jay Gambetta and John Smolin

It simulates a unitary of a quantum circuit that has been compiled to run on
the simulator. It is exponential in the number of qubits.

The input is the circuit object and the output is the same circuit object with
a result field added results['data']['unitary'] where the unitary is
a 2**n x 2**n complex numpy array representing the unitary matrix.


The input is
    compiled_circuit object
and the output is the results object

The simulator is run using

    UnitarySimulator(compiled_circuit).run().

In the qasm, key operations with type 'measure' and 'reset' are dropped.

Internal circuit_object

circuit =
    {
    'number_of_qubits': 2,
    'number_of_cbits': 2,
    'number_of_operations': 4,
    'qubit_order': {('q', 0): 0, ('v', 0): 1}
    'cbit_order': {('c', 1): 1, ('c', 0): 0},
    'qasm':
        [{
        'type': 'gate',
        'name': 'U',
        'theta': 1.570796326794897
        'phi': 1.570796326794897
        'lambda': 1.570796326794897
        'qubit_indices': [0],
        'gate_size': 1,
        },
        {
        'type': 'gate',
        'name': 'CX',
        'qubit_indices': [0, 1],
        'gate_size': 2,
        },
        {
        'type': 'reset',
        'qubit_indices': [1]
        }
        {
        'type': 'measure',
        'cbit_indices': [0],
        'qubit_indices': [0]
        }],
    }

returned results object

result =
        {
        'data':
            {
            'unitary': np.array([[ 0.70710678 +0.00000000e+00j
                                 0.70710678 -8.65956056e-17j
                                 0.00000000 +0.00000000e+00j
                                 0.00000000 +0.00000000e+00j]
                               [ 0.00000000 +0.00000000e+00j
                                 0.00000000 +0.00000000e+00j
                                 0.70710678 +0.00000000e+00j
                                 -0.70710678 +8.65956056e-17j]
                               [ 0.00000000 +0.00000000e+00j
                                 0.00000000 +0.00000000e+00j
                                 0.70710678 +0.00000000e+00j
                                 0.70710678 -8.65956056e-17j]
                               [ 0.70710678 +0.00000000e+00j
                                -0.70710678 +8.65956056e-17j
                                 0.00000000 +0.00000000e+00j
                                 0.00000000 +0.00000000e+00j]
            }
        'state': 'DONE'
        }
"""
import numpy as np
import qiskit.qasm as qasm
import qiskit.unroll as unroll
from ._simulatortools import enlarge_single_opt, enlarge_two_opt
# TODO think about if this should be an error or just removed from circuit.
# TODO add ["status"] = 'DONE', 'ERROR' especitally for empty circuit error
# does not show up


class UnitarySimulator(object):
    """Python implementation of a unitary simulator."""

    def __init__(self, compiled_circuit):
        """Initial the UnitarySimulator object."""
        basis_gates = []  # unroll to base gates
        unroller = unroll.Unroller(qasm.Qasm(data=compiled_circuit).parse(),
                                   unroll.SimulatorBackend(basis_gates))
        unroller.backend.set_trace(False)
        unroller.execute()
        self.circuit = unroller.backend.circuit
        self._number_of_qubits = self.circuit['number_of_qubits']
        self.result = {}
        self.result = {}
        self.result['data'] = {}
        self._unitary_state = np.identity(2**(self._number_of_qubits),
                                          dtype=complex)
        self._number_of_operations = self.circuit['number_of_operations']

    def _add_unitary_single(self, gate, qubit):
        """Apply the single-qubit gate.

        gate is the single-qubit gate.
        qubit is the qubit to apply it on counts from 0 and order
            is q_{n-1} ... otimes q_1 otimes q_0.
        number_of_qubits is the number of qubits in the system.
        """
        unitaty_add = enlarge_single_opt(gate, qubit, self._number_of_qubits)
        self._unitary_state = np.dot(unitaty_add, self._unitary_state)

    def _add_unitary_two(self, gate, q0, q1):
        """Apply the two-qubit gate.

        gate is the two-qubit gate
        q0 is the first qubit (control) counts from 0
        q1 is the second qubit (target)
        returns a complex numpy array
        """
        unitaty_add = enlarge_two_opt(gate, q0, q1, self._number_of_qubits)
        self._unitary_state = np.dot(unitaty_add, self._unitary_state)

    def run(self):
        """Apply the single-qubit gate."""
        for j in range(self._number_of_operations):
            # each operations
            if self.circuit['qasm'][j]['name'] == 'U':
                qubit = self.circuit['qasm'][j]['qubit_indices'][0]
                theta = self.circuit['qasm'][j]['theta']
                phi = self.circuit['qasm'][j]['phi']
                lam = self.circuit['qasm'][j]['lambda']
                gate = np.array([[np.cos(theta/2.0),
                                  -np.exp(1j*lam)*np.sin(theta/2.0)],
                                 [np.exp(1j*phi)*np.sin(theta/2.0),
                                  np.exp(1j*phi+1j*lam)*np.cos(theta/2.0)]])
                self._add_unitary_single(gate, qubit)
            elif self.circuit['qasm'][j]['name'] == 'CX':
                qubit0 = self.circuit['qasm'][j]['qubit_indices'][0]
                qubit1 = self.circuit['qasm'][j]['qubit_indices'][1]
                gate = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0],
                                 [0, 1, 0, 0]])
                self._add_unitary_two(gate, qubit0, qubit1)
            elif self.circuit['qasm'][j]['name'] == 'measure':
                print('Warning have dropped measure from unitary simulator')
            elif self.circuit['qasm'][j]['name'] == 'reset':
                print('Warning have dropped reset from unitary simulator')
            else:
                self.result['status'] = 'ERROR'
                return self.result
        self.result['data']['unitary'] = self._unitary_state
        self.result['status'] = 'DONE'
        return self.result
