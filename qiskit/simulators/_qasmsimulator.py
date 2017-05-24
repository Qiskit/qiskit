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
        'name': 'U(1.570796326794897,0.000000000000000,3.141592653589793)',
        'qubit_indices': [0],
        'gate_size': 1,
        'matrix': np.array([[ 0.70710678 +0.00000000e+00j,
                           0.70710678 -8.65956056e-17j],
                         [ 0.70710678 +0.00000000e+00j,
                          -0.70710678 +8.65956056e-17j]])
        },
        {
        'type': 'gate',
        'name': 'CX',
        'qubit_indices': [0, 1],
        'gate_size': 2,
        'matrix': np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0],
                            [0, 1, 0, 0]])
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

result =
        {
        'data':
            {
            'quantum_state': array([ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j]),
            'classical_state': 0
            }
        'status': 'DONE'
        }
"""
import numpy as np
import random
from collections import Counter
import qiskit.qasm as qasm
import qiskit.unroll as unroll

# TODO add the IF qasm operation.
# TODO add ["status"] = 'DONE', 'ERROR' especitally for empty circuit error
# does not show up


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

    def __init__(self, compiled_circuit, shots, seed=random.random()):
        """Initialize the QasmSimulator object."""
        basis_gates = []  # unroll to base gates
        unroller = unroll.Unroller(qasm.Qasm(data=compiled_circuit).parse(),
                                   unroll.SimulatorBackend(basis_gates))
        unroller.backend.set_trace(False)
        unroller.execute()
        self.circuit = unroller.backend.circuit
        self._number_of_qubits = self.circuit['number_of_qubits']
        self._number_of_cbits = self.circuit['number_of_cbits']
        self.result = {}
        self.result['data'] = {}
        self._quantum_state = 0
        self._classical_state = 0
        self._shots = shots
        random.seed(seed)
        self._number_of_operations = self.circuit['number_of_operations']

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

    def _add_qasm_two(self, gate, q0, q1):
        """Apply the two-qubit gate.

        gate is the two-qubit gate.
        q0 is the first qubit (control) counts from 0.
        q1 is the second qubit (target).
        """
        temp1 = np.zeros([1 << (self._number_of_qubits),
                          1 << (self._number_of_qubits)])
        for i in range(1 << (self._number_of_qubits - 2)):
            for j in range(2):
                for k in range(2):
                    for jj in range(2):
                        for kk in range(2):
                            temp1[self._index2(j, q0, k, q1, i),
                                  self._index2(jj, q0, kk, q1, i)] = gate[j+2*k, jj+2*kk]
        self._quantum_state = np.dot(temp1, self._quantum_state)

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
                # Check if gate operation
                if self.circuit['qasm'][j]['type'] == 'gate':
                    gate = self.circuit['qasm'][j]['matrix']
                    # Check if single or two qubit gate
                    if self.circuit['qasm'][j]['gate_size'] == 1:
                        qubit = self.circuit['qasm'][j]['qubit_indices'][0]
                        self._add_qasm_single(gate, qubit)
                    elif self.circuit['qasm'][j]['gate_size'] == 2:
                        qubit0 = self.circuit['qasm'][j]['qubit_indices'][0]
                        qubit1 = self.circuit['qasm'][j]['qubit_indices'][1]
                        # Check if CX or other two qubit gate
                        if self.circuit['qasm'][j]['name'] == 'CX':
                            self._add_qasm_cx(qubit0, qubit1)
                        else:
                            self._add_qasm_two(gate, qubit0, qubit1)
                # Check if measure
                elif self.circuit['qasm'][j]['type'] == 'measure':
                    qubit = self.circuit['qasm'][j]['qubit_indices'][0]
                    cbit = self.circuit['qasm'][j]['cbit_indices'][0]
                    self._add_qasm_measure(qubit, cbit)
                # Check if reset
                elif self.circuit['qasm'][j]['type'] == 'reset':
                    qubit = self.circuit['qasm'][j]['qubit_indices'][0]
                    self._add_qasm_reset(qubit)
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
