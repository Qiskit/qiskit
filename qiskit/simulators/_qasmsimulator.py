"""
Contains a (slow) python simulator that makes the qasm of the circuit.

We advise using the c++ simulator or online for larger size systems


{'number_of_qubits': 2,
'number_of_cbits': 2,
'number_of_operations': 2
'qubit_order': {('q', 0): 0, ('v', 0): 1}
'cbit_order': {('c', 1): 1, ('c', 0): 0},
'qasm': [{
        'type': 'gate',
        'name': 'U(1.570796326794897,0.000000000000000,3.141592653589793)',
        'qubit_indices': [0],
        'gate_size': 1,
        'matrix': array([[ 0.70710678 +0.00000000e+00j,
                           0.70710678 -8.65956056e-17j],
                         [ 0.70710678 +0.00000000e+00j,
                          -0.70710678 +8.65956056e-17j]])
        },
        {
        'type': 'measure',
        'cbit_indices': [0],
        'qubit_indices': [0]
        }],
'result': {
        'quantum_state': array([ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j]),
        'classical_state': 0
        }
}
"""
import numpy as np
import random


class QasmSimulator(object):
    """
    Python implementation of a unitary computer simulator.
    """

    def __init__(self, circuit, random_seed):
        self.circuit = circuit
        self._number_of_qubits = self.circuit['number_of_qubits']
        self._number_of_cbits = self.circuit['number_of_cbits']
        self.circuit['result'] = {}
        self._quantum_state = np.zeros(2**(self._number_of_qubits),
                                       dtype=complex)
        self._quantum_state[0] = 1
        self._classical_state = 0
        random.seed(random_seed)
        self._number_of_operations = self.circuit['number_of_operations']

    def _add_qasm_single(self, gate, qubit):
        """Apply the single qubit gate.

        gate is the single qubit gate
        qubit is the qubit to apply it on counts from 0 and order
            is q_{n-1} ... otimes q_1 otimes q_0
        number_of_qubits is the number of qubits in the system
        returns a complex numpy array
        """
        temp_1 = np.identity(2**(self._number_of_qubits-qubit-1),
                             dtype=complex)
        temp_2 = np.identity(2**(qubit), dtype=complex)
        unitaty_add = np.kron(temp_1, np.kron(gate, temp_2))
        self._quantum_state = np.dot(unitaty_add, self._quantum_state)

    def _add_qasm_two(self, gate, qubit_1, qubit_2):
        """Apply the two-qubit gate.

        gate is the two-qubit gate
        qubit_1 is the first qubit (control) counts from 0
        qubit_2 is the second qubit (target)
        number_of_qubits is the number of qubits in the system
        returns a complex numpy array
        """
        temp_1 = np.kron(np.identity(2**(self._number_of_qubits-2),
                                     dtype=complex), gate)
        unitaty_add = np.identity(2**(self._number_of_qubits), dtype=complex)
        # print('here     !!!!!!!!!!!!!!!')
        for ii in range(2**self._number_of_qubits):
            iistring = bin(ii)[2:]
            bits = list(reversed(iistring.zfill(self._number_of_qubits)))
            swap_1 = bits[0]
            swap_2 = bits[1]
            bits[0] = bits[qubit_1]
            bits[1] = bits[qubit_2]
            bits[qubit_1] = swap_1
            bits[qubit_2] = swap_2
            iistring = ''.join(reversed(bits))
            iip = int(iistring, 2)
            for jj in range(2**self._number_of_qubits):
                jjstring = bin(jj)[2:]
                bits = list(reversed(jjstring.zfill(self._number_of_qubits)))
                swap_1 = bits[0]
                swap_2 = bits[1]
                bits[0] = bits[qubit_1]
                bits[1] = bits[qubit_2]
                bits[qubit_1] = swap_1
                bits[qubit_2] = swap_2
                jjstring = ''.join(reversed(bits))
                jjp = int(jjstring, 2)
                unitaty_add[iip, jjp] = temp_1[ii, jj]
        self._quantum_state = np.dot(unitaty_add, self._quantum_state)
        # print(self._quantum_state)

    def _add_qasm_decision(self, qubit):
        """Apply the measurement/reset qubit gate."""
        # print(qubit)
        probability_zero = 0
        random_number = random.random()
        for ii in range(2**self._number_of_qubits):
            iistring = bin(ii)[2:]
            bits = list(reversed(iistring.zfill(self._number_of_qubits)))
            if bits[qubit] == '0':
                probability_zero += np.abs(self._quantum_state[ii])**2
        # print(probability_zero)
        if random_number <= probability_zero:
            outcome = '0'
            norm = np.sqrt(probability_zero)
        else:
            outcome = '1'
            norm = np.sqrt(1-probability_zero)
        return (outcome, norm)

    def _add_qasm_measure(self, qubit, cbit):
        """Apply the measurement qubit gate."""

        outcome, norm = self._add_qasm_decision(qubit)
        # print(outcome)
        # print(norm)
        for ii in range(2**self._number_of_qubits):
            # update quantum state
            iistring = bin(ii)[2:]
            bits = list(reversed(iistring.zfill(self._number_of_qubits)))
            if bits[qubit] == outcome:
                self._quantum_state[ii] = self._quantum_state[ii]/norm
            else:
                self._quantum_state[ii] = 0
        # update classical state
        temp = bin(self._classical_state)[2:]
        cbits_string = list(reversed(temp.zfill(self._number_of_cbits)))
        cbits_string[cbit] = outcome
        self._classical_state = int(''.join(reversed(cbits_string)), 2)

    def _add_qasm_reset(self, qubit):
        """Apply the reset to the qubit.

        I to this by applying a measurment and ignoring the outcome"""
        """Apply the measurement qubit gate."""

        outcome, norm = self._add_qasm_decision(qubit)
        # print(outcome)
        temp = self._quantum_state
        for ii in range(2**self._number_of_qubits):
            # update quantum state
            iistring = bin(ii)[2:]
            bits = list(reversed(iistring.zfill(self._number_of_qubits)))
            if outcome == '0':
                iip = ii
            else:
                bits[qubit] == '0'
                iip = int(''.join(reversed(bits)), 2)
            if bits[qubit] == '0':
                self._quantum_state[iip] = temp[ii]/norm
            else:
                self._quantum_state[iip] = 0
        # print(self._quantum_state)

    def run(self):
        """Run."""
        for j in range(self._number_of_operations):
            if self.circuit['qasm'][j]['type'] == 'gate':
                gate = self.circuit['qasm'][j]['matrix']
                if self.circuit['qasm'][j]['gate_size'] == 1:
                    qubit = self.circuit['qasm'][j]['qubit_indices'][0]
                    self._add_qasm_single(gate, qubit)
                elif self.circuit['qasm'][j]['gate_size'] == 2:
                    qubit0 = self.circuit['qasm'][j]['qubit_indices'][0]
                    qubit1 = self.circuit['qasm'][j]['qubit_indices'][1]
                    self._add_qasm_two(gate, qubit0, qubit1)
            elif self.circuit['qasm'][j]['type'] == 'measure':
                qubit = self.circuit['qasm'][j]['qubit_indices'][0]
                cbit = self.circuit['qasm'][j]['cbit_indices'][0]
                self._add_qasm_measure(qubit, cbit)
            elif self.circuit['qasm'][j]['type'] == 'reset':
                qubit = self.circuit['qasm'][j]['qubit_indices'][0]
                self._add_qasm_reset(qubit)
        self.circuit['result']['quantum_state'] = self._quantum_state
        self.circuit['result']['classical_state'] = self._classical_state
        return self.circuit
