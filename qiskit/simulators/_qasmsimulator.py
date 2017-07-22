# pylint: disable=line-too-long
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

"""Contains a (slow) python simulator.

Author: Jay Gambetta and John Smolin

It simulates a qasm quantum circuit that has been compiled to run on the
simulator. It is exponential in the number of qubits.

We advise using the c++ simulator or online for larger size systems.

The input is
    compiled_circuit object
    shots
    seed
and the output is the results object

if shots = 1
    compiled_circuit['result']['data']['quantum_state'] and
    results['data']['classical_state'] where quantum_state is
a 2**n complex numpy array representing the quantum state vector and
classical_state is a interger representing the state of the classical
registors.
if shots > 1
    results['data']["counts"] where this is dict {"0000" : 454}

The simulator is run using

    QasmSimulator(compiled_circuit,shots,seed).run().

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
            "qubits": , // optional -- list[int]
            "clbits": , //optional -- list[int]
            "conditional":  // optional -- map
                {
                    "type": , // string
                    "mask": , // big int
                    "val":  , // big int
                }
        },
    ]
}

if shots = 1
result =
        {
        'data':
            {
            'quantum_state': array([ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j]),
            'classical_state': 0
            }
        'status': 'DONE'
        }

if shots > 1
result =
        {
        'data':
            {
            'counts': {'0000': 50, '1001': 44},
            }
        'status': 'DONE'
        }
"""
import numpy as np
import random
from collections import Counter
import json


# TODO add the IF qasm operation.
# TODO add ["status"] = 'DONE', 'ERROR' especitally for empty circuit error
# does not show up

__configuration = {"name": "local_qasm_simulator",
                   "url": "https://github.com/IBM/qiskit-sdk-py",
                   "simulator": True,
                   "description": "A python simulator for qasm files",
                   "nQubits": 10,
                   "couplingMap": "all-to-all",
                   "gateset": "SU2+CNOT"}


class QasmSimulator(object):
    """Python implementation of a qasm simulator."""

    @staticmethod
    def _index1(b, i, k):
        """Magic index1 function.

        Takes a bitstring k and inserts bit b as the ith bit,
        shifting bits >= i over to make room.
        """
        retval = k
        lowbits = k & ((1 << i) - 1)  # get the low i bits

        retval >>= i
        retval <<= 1

        retval |= b

        retval <<= i
        retval |= lowbits

        return retval

    @staticmethod
    def _index2(b1, i1, b2, i2, k):
        """Magic index1 function.

        Takes a bitstring k and inserts bits b1 as the i1th bit
        and b2 as the i2th bit
        """
        assert(i1 != i2)

        if i1 > i2:
            # insert as (i1-1)th bit, will be shifted left 1 by next line
            retval = QasmSimulator._index1(b1, i1-1, k)
            retval = QasmSimulator._index1(b2, i2, retval)
        else:  # i2>i1
            # insert as (i2-1)th bit, will be shifted left 1 by next line
            retval = QasmSimulator._index1(b2, i2-1, k)
            retval = QasmSimulator._index1(b1, i1, retval)
        return retval

    def __init__(self, job):
        """Initialize the QasmSimulator object."""
        self.circuit = json.loads(job['compiled_circuit'])
        self._number_of_qubits = self.circuit['header']['number_of_qubits']
        self._number_of_cbits = self.circuit['header']['number_of_clbits']
        self.result = {}
        self.result['data'] = {}
        self._quantum_state = 0
        self._classical_state = 0
        self._shots = job['config']['shots']
        random.seed(job['config']['seed'])
        self._number_of_operations = len(self.circuit['operations'])

    def _add_qasm_single(self, gate, qubit):
        """Apply an arbitary 1-qubit operator to a qubit.

        Gate is the single qubit applied.
        qubit is the qubit the gate is applied to.
        """
        psi = self._quantum_state
        bit = 1 << qubit
        for k1 in range(0, 1 << self._number_of_qubits, 1 << (qubit+1)):
            for k2 in range(0, 1 << qubit, 1):
                k = k1 | k2
                cache0 = psi[k]
                cache1 = psi[k | bit]
                psi[k] = gate[0, 0] * cache0 + gate[0, 1] * cache1
                psi[k | bit] = gate[1, 0] * cache0 + gate[1, 1] * cache1

    def _add_qasm_cx(self, q0, q1):
        """Optimized ideal CX on two qubits.

        q0 is the first qubit (control) counts from 0.
        q1 is the second qubit (target).
        """
        psi = self._quantum_state
        for k in range(0, 1 << (self._number_of_qubits - 2)):
            # first bit is control, second is target
            ind1 = self._index2(1, q0, 0, q1, k)
            # swap target if control is 1
            ind3 = self._index2(1, q0, 1, q1, k)
            cache0 = psi[ind1]
            cache1 = psi[ind3]
            psi[ind3] = cache0
            psi[ind1] = cache1

    def _add_qasm_decision(self, qubit):
        """Apply the decision of measurement/reset qubit gate.

        qubit is the qubit that is measured/reset
        """
        probability_zero = 0
        random_number = random.random()
        for ii in range(1 << self._number_of_qubits):
            if ii & (1 << qubit) == 0:
                probability_zero += np.abs(self._quantum_state[ii])**2
        if random_number <= probability_zero:
            outcome = '0'
            norm = np.sqrt(probability_zero)
        else:
            outcome = '1'
            norm = np.sqrt(1-probability_zero)
        return (outcome, norm)

    def _add_qasm_measure(self, qubit, cbit):
        """Apply the measurement qubit gate.

        qubit is the qubit measured.
        cbit is the classical bit the measurement is assigned to.
        """
        outcome, norm = self._add_qasm_decision(qubit)
        for ii in range(1 << self._number_of_qubits):
            # update quantum state
            if (ii >> qubit) & 1 == int(outcome):
                self._quantum_state[ii] = self._quantum_state[ii]/norm
            else:
                self._quantum_state[ii] = 0
        # update classical state
        bit = 1 << cbit
        self._classical_state = (self._classical_state & (~bit)) | (int(outcome) << cbit)

    def _add_qasm_reset(self, qubit):
        """Apply the reset to the qubit.

        This is done by doing a measruement and if 0 do nothing and
        if 1 flip the qubit.

        qubit is the qubit that is reset.
        """
        # TODO: slow, refactor later
        outcome, norm = self._add_qasm_decision(qubit)
        temp = np.copy(self._quantum_state)
        self._quantum_state.fill(0.0)
        # measurement
        for ii in range(1 << self._number_of_qubits):
            if (ii >> qubit) & 1 == int(outcome):
                temp[ii] = temp[ii]/norm
            else:
                temp[ii] = 0
        # reset
        if outcome == '1':
            for ii in range(1 << self._number_of_qubits):
                iip = (~ (1 << qubit)) & ii  # bit number qubit set to zero
                self._quantum_state[iip] += temp[ii]
        else:
            self._quantum_state = temp

    def _qasm_single_params(self, gate, params=None):
        """Apply a single qubit gate to the qubit.

        Args:
            gate(str): the single qubit gate name
            params(list): the operation parameters op['params']
        Returns:
            a tuple of U gate parameters (theta, phi, lam)
        """
        if gate == 'U' or gate == 'u3':
            return (params[0], params[1], params[2])
        elif gate == 'u2':
            return (np.pi/2, params[0], params[1])
        elif gate == 'u1':
            return (0., 0., params[0])

    def run(self):
        """Run."""
        outcomes = []
        # Do each shot
        for shot in range(self._shots):
            self._quantum_state = np.zeros(1 << self._number_of_qubits, dtype=complex)
            self._quantum_state[0] = 1
            self._classical_state = 0
            # Do each operation in this shot
            for j in range(self._number_of_operations):
                # get operation name
                op = self.circuit['operations'][j]
                # Check if single  gate
                if op["name"] in ['U', 'u1', 'u2', 'u3']:
                    if 'params' in op:
                        params = op['params']
                    else:
                        params = None
                    (theta, phi, lam) = self._qasm_single_params(op['name'], params)
                    qubit = op['qubits'][0]
                    gate = np.array([[np.cos(theta/2.0),
                                      -np.exp(1j*lam)*np.sin(theta/2.0)],
                                     [np.exp(1j*phi)*np.sin(theta/2.0),
                                      np.exp(1j*phi+1j*lam)*np.cos(theta/2.0)]])
                    self._add_qasm_single(gate, qubit)
                # Check if CX gate
                elif op['name'] == 'CX' or op['name'] == 'cx':
                    qubit0 = op['qubits'][0]
                    qubit1 = op['qubits'][1]
                    self._add_qasm_cx(qubit0, qubit1)
                # Check if measure
                elif op['name'] == 'measure':
                    qubit = op['qubits'][0]
                    cbit = op['clbits'][0]
                    self._add_qasm_measure(qubit, cbit)
                # Check if reset
                elif op['name'] == 'reset':
                    qubit = op['qubits'][0]
                    self._add_qasm_reset(qubit)
                elif op['name'] == 'barrier' or op['name'] == 'id':
                    pass
                else:
                    self.result['status'] = 'ERROR'
                    return self.result
            # Turn classical_state (int) into bit string
            outcomes.append(bin(self._classical_state)[2:].zfill(self._number_of_cbits))
        # Return the results
        if self._shots == 1:
            self.result['data']['quantum_state'] = self._quantum_state
            self.result['data']['classical_state'] = self._classical_state

        else:
            self.result['data']['counts'] = dict(Counter(outcomes))
        self.result['status'] = 'DONE'
        return self.result
