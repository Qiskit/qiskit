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

It produces the unitary of a quantum circuit in the symbolic form.
In particular, it simulates the quantum computation with the sympy APIs,
which preserve the symbolic form of numbers, e.g., sqrt(2), e^{i*pi/2}.

example code for using it:
Q_program = QuantumProgram()
currentFolder = os.path.dirname(os.path.realpath(__file__))
qasm_file = currentFolder + "/../../../examples/qasm/naive.qasm"
myqasm = Q_program.load_qasm_file(qasm_file, "my_example")
print("analyzing: " + qasm_file)
circuits = ['my_example'] #, 'superposition'
backend = 'local_sympy_unitary_simulator' # the device to run on
result = Q_program.execute(circuits, backend=backend, timeout=10)
print("unitary matrix of the circuit: ")
print(result.get_data('my_example')['unitary'])

example output:
[[sqrt(2)/2 sqrt(2)/2 0 0]
 [0 0 sqrt(2)/2 -sqrt(2)/2]
 [0 0 sqrt(2)/2 sqrt(2)/2]
 [sqrt(2)/2 -sqrt(2)/2 0 0]]

Warning: it is slow.



"""
import uuid
import logging
import numpy as np
import json
from qiskit._result import Result
from qiskit.backends._basebackend import BaseBackend
from sympy.physics.quantum import TensorProduct

from sympy.matrices import eye, zeros

from sympy.physics.quantum.qubit import Qubit
from sympy.physics.quantum.gate import H, X, Y, Z, S, T, CNOT, IdentityGate, OneQubitGate, UGate
from sympy.core.compatibility import is_sequence, u, unicode, range
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.represent import represent
from sympy import pprint, pretty, Matrix, Integer, I, pi, E, Pow, exp, log, Add, sqrt, Mul
from sympy.physics.quantum.qubit import IntQubit, matrix_to_qubit

from sympy.physics.quantum.qubit import measure_partial, qubit_to_matrix
from sympy import conjugate, N, re, im
from sympy.physics.quantum import TensorProduct

from sympy.physics.quantum.gate import u, H, X, Y, Z, S, T, CNOT, IdentityGate, OneQubitGate, TwoQubitGate, Gate, XGate, CGate, UGate
from sympy.physics.quantum.represent import represent

from sympy import pprint, pretty, symbols, Matrix, pi, E, I, cos, sin, N, exp, nsimplify

logger = logging.getLogger(__name__)


# TODO add ["status"] = 'DONE', 'ERROR' especitally for empty circuit error
# does not show up

def index1(b, i, k):
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

def index2(b1, i1, b2, i2, k):
    """Magic index1 function.

    Takes a bitstring k and inserts bits b1 as the i1th bit
    and b2 as the i2th bit
    """
    assert(i1 != i2)

    if i1 > i2:
        # insert as (i1-1)th bit, will be shifted left 1 by next line
        retval = index1(b1, i1-1, k)
        retval = index1(b2, i2, retval)
    else:  # i2>i1
        # insert as (i2-1)th bit, will be shifted left 1 by next line
        retval = index1(b2, i2-1, k)
        retval = index1(b1, i1, retval)
    return retval


class SympyUnitarySimulator(BaseBackend):
    """Python implementation of a unitary simulator."""

    def __init__(self, configuration=None):
        """Initial the UnitarySimulator object."""
        if configuration is None:
            self._configuration = {'name': 'local_sympy_unitary_simulator',
                                   'url': 'https://github.com/IBM/qiskit-sdk-py',
                                   'simulator': True,
                                   'local': True,
                                   'description': 'A python simulator for unitary matrix',
                                   'coupling_map': 'all-to-all',
                                   'basis_gates': 'u1,u2,u3,cx,id'}
        else:
            self._configuration = configuration


    # the following is the sympy implementations of various gates, including basic gates and ugates.
    @staticmethod
    def regulate(theta):
        error_margin = 0.01

        if abs(N(theta - pi)) < error_margin:
            return pi, True
        elif abs(N(theta - pi/2)) < error_margin:
            return pi/2, True
        elif abs(N(theta - pi/4)) < error_margin:
            return pi/4, True
        elif abs(N(theta- 2*pi)) < error_margin:
            return 2*pi, True
        else:
            return theta, theta == 0 # if theta ==0, we also think it is regular




    @staticmethod
    def compute_ugate_matrix(parafloatlist):
        theta = parafloatlist[0]
        phi = parafloatlist[1]
        lamb = parafloatlist[2]

        theta, theta_is_regular = SympyUnitarySimulator.regulate(theta)
        phi, phi_is_regular = SympyUnitarySimulator.regulate(phi)
        lamb, lamb_is_regular = SympyUnitarySimulator.regulate(lamb)


        uMat = Matrix([[cos(theta/2), (-E**(I*lamb)) * sin(theta/2)],
                       [(E**(I*phi)) * sin(theta/2), (E**(I*(phi+lamb))) * cos(theta/2)]])

        if theta_is_regular and phi_is_regular and lamb_is_regular: # regular: we do not need concrete float value
            uMatNumeric = uMat
        else:
            uMatNumeric = uMat.evalf()
        return uMatNumeric


    @staticmethod
    def compute_ugate_matrix_wrap(parafloatlist):
        if len(parafloatlist) == 1: # [theta=0, phi=0, lambda]
            parafloatlist.insert(0, 0.0)
            parafloatlist.insert(0, 0.0)
        elif len(parafloatlist) == 2: #[theta=pi/2, phi, lambda]
            parafloatlist.insert(0, pi/2)
        elif len(parafloatlist) == 3: #[theta, phi, lambda]
            pass
        else:
            return NotImplemented

        uMat = SympyUnitarySimulator.compute_ugate_matrix(parafloatlist)
        return uMat


    def enlarge_single_opt_sympy(self, opt, qubit, number_of_qubits):
        """Enlarge single operator to n qubits.

        It is exponential in the number of qubits.

        Args:
            opt: the single-qubit opt.
            qubit: the qubit to apply it on counts from 0 and order
                is q_{n-1} ... otimes q_1 otimes q_0.
            number_of_qubits: the number of qubits in the system.
        """
        temp_1 = eye(2**(number_of_qubits-qubit-1))
        temp_2 = eye(2**(qubit))
        enlarge_opt = TensorProduct(temp_1, TensorProduct(opt, temp_2)) # peng: smart trick!
        return enlarge_opt


    def _add_unitary_single(self, gate, qubit):
        """Apply the single-qubit gate.

        gate is the single-qubit gate.
        qubit is the qubit to apply it on counts from 0 and order
            is q_{n-1} ... otimes q_1 otimes q_0.
        number_of_qubits is the number of qubits in the system.
        """
        unitaty_add = self.enlarge_single_opt_sympy(gate, qubit, self._number_of_qubits)
        self._unitary_state = unitaty_add*self._unitary_state # * means "dot product"



    def enlarge_two_opt_sympy(self,opt, q0, q1, num):
        """Enlarge two-qubit operator to n qubits.

        It is exponential in the number of qubits.
        opt is the two-qubit gate
        q0 is the first qubit (control) counts from 0
        q1 is the second qubit (target)
        returns a complex numpy array
        number_of_qubits is the number of qubits in the system.
        """
        enlarge_opt = zeros(1 << (num), 1 << (num)) # np.zeros([1 << (num), 1 << (num)])
        for i in range(1 << (num-2)):
            for j in range(2):
                for k in range(2):
                    for jj in range(2):
                        for kk in range(2):
                            enlarge_opt[index2(j, q0, k, q1, i), index2(jj, q0, kk, q1, i)] = opt[j+2*k, jj+2*kk]
        return enlarge_opt

    def _add_unitary_two(self, gate, q0, q1):
        """Apply the two-qubit gate.

        gate is the two-qubit gate
        q0 is the first qubit (control) counts from 0
        q1 is the second qubit (target)
        returns a complex numpy array
        """
        unitaty_add = self.enlarge_two_opt_sympy(gate, q0, q1, self._number_of_qubits)
        self._unitary_state = unitaty_add*self._unitary_state

    def run(self, q_job):
        """Run q_job

        Args:
        q_job (QuantumJob): job to run
        """
        # Generating a string id for the job
        job_id = str(uuid.uuid4())
        qobj = q_job.qobj
        result_list = []
        for circuit in qobj['circuits']:
            result_list.append( self.run_circuit(circuit) )
        return Result({'job_id': job_id, 'result': result_list, 'status': 'COMPLETED'},
                      qobj)            

    def run_circuit(self, circuit):
        """Apply the single-qubit gate."""
        ccircuit = circuit['compiled_circuit']
        self._number_of_qubits = ccircuit['header']['number_of_qubits']
        result = {}
        result['data'] = {}
        self._unitary_state = eye(2**(self._number_of_qubits))
        for operation in ccircuit['operations']:
            if operation['name'] in ['U', 'u1', 'u2', 'u3']:
                if 'params' in operation:
                    params = operation['params']
                else:
                    params = None
                qubit = operation['qubits'][0]
                gate = SympyUnitarySimulator.compute_ugate_matrix_wrap(params)
                self._add_unitary_single(gate, qubit)
            elif operation['name'] in ['id', 'u0']:
                pass
            elif operation['name'] in ['CX', 'cx']:
                qubit0 = operation['qubits'][0]
                qubit1 = operation['qubits'][1]
                gate = Matrix([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
                #np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0],
                #                 [0, 1, 0, 0]])
                self._add_unitary_two(gate, qubit0, qubit1)
            elif operation['name'] == 'measure':
                logger.info('Warning have dropped measure from unitary '
                            'simulator')
            elif operation['name'] == 'reset':
                logger.info('Warning have dropped reset from unitary '
                            'simulator')
            elif operation['name'] == 'barrier':
                pass
            else:
                result['status'] = 'ERROR'
                return result
        result['data']['unitary'] = np.array(self._unitary_state)
        result['status'] = 'DONE'
        return result
