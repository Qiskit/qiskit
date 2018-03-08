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

How to use this simulator:
see examples/python/use_sympy_simulators.py

Example output:
[[sqrt(2)/2 sqrt(2)/2 0 0]
 [0 0 sqrt(2)/2 -sqrt(2)/2]
 [0 0 sqrt(2)/2 sqrt(2)/2]
 [sqrt(2)/2 -sqrt(2)/2 0 0]]

Warning: it is slow.
"""
import uuid
import logging
import numpy as np
from sympy.matrices import eye, zeros
from sympy.physics.quantum import TensorProduct
from sympy import Matrix, pi, E, I, cos, sin, N
from qiskit._result import Result
from qiskit.backends._basebackend import BaseBackend

logger = logging.getLogger(__name__)


def index1(barg, iarg, karg):
    """Magic index1 function.

    Takes a bitstring k and inserts bit b as the ith bit,
    shifting bits >= i over to make room.
    """
    retval = karg
    lowbits = karg & ((1 << iarg) - 1)  # get the low i bits

    retval >>= iarg
    retval <<= 1

    retval |= barg

    retval <<= iarg
    retval |= lowbits

    return retval


def index2(barg1, iarg1, barg2, iarg2, k):  # assert(i1 != i2)
    """Magic index1 function.

    Takes a bitstring k and inserts bits b1 as the i1th bit
    and b2 as the i2th bit
    """
    if iarg1 > iarg2:
        # insert as (i1-1)th bit, will be shifted left 1 by next line
        retval = index1(barg1, iarg1 - 1, k)
        retval = index1(barg2, iarg2, retval)
    else:  # i2>i1
        # insert as (i2-1)th bit, will be shifted left 1 by next line
        retval = index1(barg2, iarg2 - 1, k)
        retval = index1(barg1, iarg1, retval)
    return retval


class SympyUnitarySimulator(BaseBackend):
    """Python implementation of a unitary simulator."""

    def __init__(self, configuration=None):
        """Initial the UnitarySimulator object."""
        super(SympyUnitarySimulator, self).__init__()
        if configuration is None:
            self._configuration = {'name': 'local_sympy_unitary_simulator',
                                   'url': 'https://github.com/QISKit/qiskit-sdk-py',
                                   'simulator': True,
                                   'local': True,
                                   'description': 'A python simulator for unitary matrix',
                                   'coupling_map': 'all-to-all',
                                   'basis_gates': 'u1,u2,u3,cx,id'}
        else:
            self._configuration = configuration
        self._unitary_state = None
        self._number_of_qubits = None

    # the following is the sympy implementations of various gates, including basic gates and ugates.
    @staticmethod
    def regulate(theta):
        """Find the most close symbolic form"""
        error_margin = 0.01

        if abs(N(theta - pi)) < error_margin:
            return pi, True
        if abs(N(theta - pi/2)) < error_margin:
            return pi/2, True
        if abs(N(theta - pi/4)) < error_margin:
            return pi/4, True
        if abs(N(theta - 2*pi)) < error_margin:
            return 2*pi, True

        return theta, theta == 0  # if theta ==0, we also think it is regular

    @staticmethod
    def compute_ugate_matrix(parafloatlist):
        """compute the ugate matrix
            Args:
                parafloatlist(list): parafloatlist
            Returns:
                Matrix: the matrix that represents the U gate
        """
        theta = parafloatlist[0]
        phi = parafloatlist[1]
        lamb = parafloatlist[2]

        theta, theta_is_regular = SympyUnitarySimulator.regulate(theta)
        phi, phi_is_regular = SympyUnitarySimulator.regulate(phi)
        lamb, lamb_is_regular = SympyUnitarySimulator.regulate(lamb)
        left_up = cos(theta/2)
        right_up = (-E**(I*lamb)) * sin(theta/2)
        left_down = (E**(I*phi)) * sin(theta/2)
        right_down = (E**(I*(phi+lamb))) * cos(theta/2)
        u_mat = Matrix([[left_up, right_up], [left_down, right_down]])

        if theta_is_regular and phi_is_regular and lamb_is_regular:
            u_mat_numeric = u_mat
        else:
            u_mat_numeric = u_mat.evalf()
        return u_mat_numeric

    @staticmethod
    def compute_ugate_matrix_wrap(parafloatlist):
        """preparation needed before computing the ugate matrix
            Args:
                parafloatlist (list): parafloatlist
            Returns:
                Matrix: matrix that represents the ugate
        """
        if len(parafloatlist) == 1:  # [theta=0, phi=0, lambda]
            parafloatlist.insert(0, 0.0)
            parafloatlist.insert(0, 0.0)
        elif len(parafloatlist) == 2:  # [theta=pi/2, phi, lambda]
            parafloatlist.insert(0, pi/2)
        elif len(parafloatlist) == 3:  # [theta, phi, lambda]
            pass
        else:
            return NotImplemented

        u_mat = SympyUnitarySimulator.compute_ugate_matrix(parafloatlist)
        return u_mat

    def enlarge_single_opt_sympy(self, opt, qubit, number_of_qubits):
        """Enlarge single operator to n qubits.

        It is exponential in the number of qubits.

        Args:
            opt (object): the single-qubit opt.
            qubit (int): the qubit to apply it on counts from 0 and order
                is q_{n-1} ... otimes q_1 otimes q_0.
            number_of_qubits (int): the number of qubits in the system.
        Returns:
            object: the enlarged tensor product.
        """
        temp_1 = eye(2**(number_of_qubits-qubit-1))
        temp_2 = eye(2**(qubit))
        enlarge_opt = TensorProduct(temp_1, TensorProduct(opt, temp_2))
        return enlarge_opt

    def _add_unitary_single(self, gate, qubit):
        """Apply the single-qubit gate.

        gate is the single-qubit gate.
        qubit is the qubit to apply it on counts from 0 and order
            is q_{n-1} ... otimes q_1 otimes q_0.
        number_of_qubits is the number of qubits in the system.
        """
        unitaty_add = self.enlarge_single_opt_sympy(gate, qubit, self._number_of_qubits)
        self._unitary_state = unitaty_add*self._unitary_state  # * means "dot product"

    def enlarge_two_opt_sympy(self, opt, qubit0, qubit1, num):
        """Enlarge two-qubit operator to n qubits.
        It is exponential in the number of qubits.
        opt is the two-qubit gate
        q0 is the first qubit (control) counts from 0
        q1 is the second qubit (target)
        returns a complex numpy array
        number_of_qubits is the number of qubits in the system.
        """
        enlarge_opt = zeros(1 << (num), 1 << (num))  # np.zeros([1 << (num), 1 << (num)])
        for i in range(1 << (num-2)):
            for j in range(2):
                for k in range(2):
                    for m in range(2):
                        for n in range(2):
                            enlarge_index1 = index2(j, qubit0, k, qubit1, i)
                            enlarge_index2 = index2(m, qubit0, n, qubit1, i)
                            enlarge_opt[enlarge_index1, enlarge_index2] = opt[j+2*k, m+2*n]
        return enlarge_opt

    def _add_unitary_two(self, gate, qubit0, qubit1):
        """Apply the two-qubit gate.

        gate is the two-qubit gate
        q0 is the first qubit (control) counts from 0
        q1 is the second qubit (target)
        returns a complex numpy array
        """
        unitaty_add = self.enlarge_two_opt_sympy(gate, qubit0, qubit1, self._number_of_qubits)
        self._unitary_state = unitaty_add*self._unitary_state

    def run(self, q_job):
        """Run q_job

        Args:
            q_job (QuantumJob): job to run
        Returns:
            object: a dictionary that contains the necessary information is returned
        """
        # Generating a string id for the job
        job_id = str(uuid.uuid4())
        qobj = q_job.qobj
        result_list = []
        for circuit in qobj['circuits']:
            result_list.append(self.run_circuit(circuit))
        return Result({'job_id': job_id, 'result': result_list, 'status': 'COMPLETED'}, qobj)

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
