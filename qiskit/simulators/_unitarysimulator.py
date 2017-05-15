"""
Contains a (slow) python simulator that makes the unitary of the circuit.

These are simple methods for making common matrices used in quantum computing.

Author: Jay Gambetta

    circuit =
        {
        'number_of_qubits': 2,
        'number_of_operations': 2
        'qubit_order': {('q', 0): 0, ('v', 0): 1}
        'qasm': [{
            'type': 'gate',
            'name': 'U(1.570796326794897,0.000000000000000,3.141592653589793)',
            'qubit_indices': [0],
            'gate_size': 1,
            'matrix': array([[ 0.70710678 +0.00000000e+00j,
                               0.70710678 -8.65956056e-17j],
                             [ 0.70710678 +0.00000000e+00j,
                              -0.70710678 +8.65956056e-17j]])
            }]
        'result': {
            unitary: array([[ 0.70710678 +0.00000000e+00j,
                              0.70710678 -8.65956056e-17j],
                            [ 0.70710678 +0.00000000e+00j,
                             -0.70710678 +8.65956056e-17j]])
            }
        }
"""
import numpy as np


class UnitarySimulator(object):
    """UnitarySimulator class."""

    def __init__(self, circuit):
        self.circuit = circuit
        self._number_of_qubits = self.circuit['number_of_qubits']
        self.circuit['result'] = {}
        self._unitary_state = np.identity(2**(self._number_of_qubits),
                                          dtype=complex)
        self._number_of_operations = self.circuit['number_of_operations']
        # print(self._unitary_state)

    def _add_unitary_single(self, gate, qubit):
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
        self._unitary_state = np.dot(unitaty_add, self._unitary_state)

    def _add_unitary_two(self, gate, qubit_1, qubit_2):
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
        for ii in range(2**self._number_of_qubits):
            iistring = bin(ii)[2:]
            bits = list(reversed(iistring.zfill(self._number_of_qubits)))
            # this is reverse order for assignement as list jut put it out
            # 0001 is 4 qubit int 1 and this becomes [0,0,0,1]
            # print(bits)
            swap_1 = bits[0]
            swap_2 = bits[1]
            bits[0] = bits[qubit_1]
            bits[1] = bits[qubit_2]
            bits[qubit_1] = swap_1
            bits[qubit_2] = swap_2
            iistring = ''.join(reversed(bits))
            iip = int(iistring, 2)
            # print(iip)
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
        self._unitary_state = np.dot(unitaty_add, self._unitary_state)

    def run(self):
        """Apply the single qubit gate."""
        for j in range(self._number_of_operations):
            if self.circuit['qasm'][j]['type'] == 'gate':
                gate = self.circuit['qasm'][j]['matrix']
                if self.circuit['qasm'][j]['gate_size'] == 1:
                    qubit = self.circuit['qasm'][j]['qubit_indices'][0]
                    self._add_unitary_single(gate, qubit)
                elif self.circuit['qasm'][j]['gate_size'] == 2:
                    qubit0 = self.circuit['qasm'][j]['qubit_indices'][0]
                    qubit1 = self.circuit['qasm'][j]['qubit_indices'][1]
                    self._add_unitary_two(gate, qubit0, qubit1)
            elif self.circuit['qasm'][j]['type'] == 'measure':
                print('Warning have droped measure from unitary simulator')
            elif self.circuit['qasm'][j]['type'] == 'reset':
                print('Warning have droped reset from unitary simulator')
        self.circuit['result']['unitary'] = self._unitary_state
        return self.circuit
