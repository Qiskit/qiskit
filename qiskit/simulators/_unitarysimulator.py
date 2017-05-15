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
            }],
        'result':
            { 'data':
                {
                unitary: array([[ 0.70710678 +0.00000000e+00j,
                                  0.70710678 -8.65956056e-17j],
                                [ 0.70710678 +0.00000000e+00j,
                                 -0.70710678 +8.65956056e-17j]])
                }
            }
        }
"""
import numpy as np


class UnitarySimulator(object):
    """UnitarySimulator class."""

    @staticmethod
    def _index1(b,i,k):
        "takes a bitstring k and inserts bit b as the ith bit,shifting bits >= i over to make room"

        retval=k
        lowbits=k & ( (1<<i) - 1)  # get the low i bits

        retval >>= i
        retval <<= 1

        retval |= b

        retval <<= i
        retval |= lowbits

        return retval
    #-------------------------------------------------------------
    @staticmethod
    def _index2(b1,i1,b2,i2,k):
        "takes a bitstring k and inserts bits b1 as the i1th bit and b2 as the i2th bit"

        assert(i1 != i2)

        if i1 > i2:
            retval = UnitarySimulator._index1(b1,i1-1,k) # insert as (i1-1)th bit, will be shifted left 1 by next line
            retval = UnitarySimulator._index1(b2,i2,retval)
        else:  # i2>i1
            retval = UnitarySimulator._index1(b2,i2-1,k) # insert as (i2-1)th bit, will be shifted left 1 by next line
            retval = UnitarySimulator._index1(b1,i1,retval)
        return retval
    #-------------------------------------------------------------


    def __init__(self, circuit):
        self.circuit = circuit
        self._number_of_qubits = self.circuit['number_of_qubits']
        self.circuit['result'] = {}
        self.circuit['result']['data'] = {}
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

    def _add_unitary_two(self,gate,q0,q1):
        """Apply the two-qubit gate.
        gate is the two-qubit gate
        q0 is the first qubit (control) counts from 0
        q1 is the second qubit (target)
        returns a complex numpy array
        """
        temp1=np.zeros([1<<(self._number_of_qubits),1<<(self._number_of_qubits)])
        for i in range(1<<(self._number_of_qubits-2)):
            for j in range(2):
                for k in range(2):
                    for jj in range(2):
                        for kk in range(2):
                            temp1[self._index2(j,q0,k,q1,i), self._index2(jj,q0,kk,q1,i)]=gate[j+2*k,jj+2*kk]
        self._unitary_state = np.dot(temp1, self._unitary_state)

    def _add_unitary_two_broken(self, gate, qubit_1, qubit_2):
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
                print('Warning have dropped measure from unitary simulator')
            elif self.circuit['qasm'][j]['type'] == 'reset':
                print('Warning have dropped reset from unitary simulator')
        self.circuit['result']['data']['unitary'] = self._unitary_state
        return self.circuit
