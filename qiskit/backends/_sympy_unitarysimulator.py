# -*- coding: utf-8 -*-

# Copyright 2018 IBM RESEARCH. All Rights Reserved.
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
import logging
import uuid

import numpy as np
from sympy import Matrix, pi
from sympy.matrices import eye, zeros
from sympy.physics.quantum import TensorProduct

from qiskit._result import Result
from qiskit.backends._basebackend import BaseBackend
from qiskit.backends._simulatortools import compute_ugate_matrix, index2

logger = logging.getLogger(__name__)


class SympyUnitarySimulator(BaseBackend):
    """Python implementation of a unitary simulator."""

    def __init__(self, configuration=None):
        """Initial the UnitarySimulator object."""
        super().__init__(configuration)
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

    @staticmethod
    def compute_ugate_matrix_wrap(parameters):
        """
            convert the parameter lists used by U1, U2 to the same form as U3.
            then computes the matrix for the u gate based on the parameter list
            Args:
                parameters (list): list of parameters, of which the length may be 1, 2, or 3
            Returns:
                Matrix: the matrix that represents the ugate
        """
        if len(parameters) == 1:  # [theta=0, phi=0, lambda]
            parameters.insert(0, 0.0)
            parameters.insert(0, 0.0)
        elif len(parameters) == 2:  # [theta=pi/2, phi, lambda]
            parameters.insert(0, pi / 2)
        elif len(parameters) == 3:  # [theta, phi, lambda]
            pass
        else:
            return NotImplemented

        u_mat = compute_ugate_matrix(parameters)
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
            Matrix: the enlarged matrix that operates on all qubits in the system.
        """
        temp_1 = eye(2**(number_of_qubits-qubit-1))
        temp_2 = eye(2**(qubit))
        enlarge_opt = TensorProduct(temp_1, TensorProduct(opt, temp_2))
        return enlarge_opt

    def _add_unitary_single(self, gate, qubit):
        """Apply the single-qubit gate.
        Args:
            gate (Matrix): The matrix for a single-qubit gate. It looks like this:
                        Matrix([
                            [sqrt(2)/2,  sqrt(2)/2],
                            [sqrt(2)/2, -sqrt(2)/2]])
                        Matrix is a type from sympy.
            qubit (int): the id of the qubit being operated on
        """
        unitaty_add = self.enlarge_single_opt_sympy(gate, qubit, self._number_of_qubits)
        self._unitary_state = unitaty_add*self._unitary_state  # * means "dot product"

    def enlarge_two_opt_sympy(self, opt, qubit0, qubit1, num):
        """Enlarge two-qubit operator to n qubits.
        It is exponential in the number of qubits.
        Args:
            opt (Matrix): the matrix that represents a two-qubit gate.
            It looks like this:
                    Matrix([
                            [1, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0]])
            qubit0 (int): id of the control qubit
            qubit1 (int): id of the target qubit
            num (int): the number of qubits in the system.
        Returns:
            Matrix: the enlarged matrix that operates on all qubits in the system.
                    It is basically a tensorproduct of the gates applied on each qubit
                    (Identity gate if no gate is applied to the qubit).
        """
        enlarge_opt = zeros(2**num, 2**num)  # np.zeros([1 << (num), 1 << (num)])
        for i in range(2**(num-2)):
            for j in range(2):
                for k in range(2):
                    for m in range(2):
                        for n in range(2):
                            enlarge_index1 = index2(j, qubit0, k, qubit1, i)
                            enlarge_index2 = index2(m, qubit0, n, qubit1, i)
                            enlarge_opt[enlarge_index1, enlarge_index2] = opt[j+2*k, m+2*n]
        return enlarge_opt

    def _add_unitary_two(self, gate, qubit0, qubit1):
        """Apply the two-qubit gate
         It first extends the two-qubit gate to all-qubit gate and then applying it to all qubits.
         The result stored in self.__unitary_state is a unitary matrix, which looks like this:
                    Matrix([
                        [sqrt(2)/2,  sqrt(2)/2,         0,          0],
                        [        0,          0, sqrt(2)/2, -sqrt(2)/2],
                        [        0,          0, sqrt(2)/2,  sqrt(2)/2],
                        [sqrt(2)/2, -sqrt(2)/2,         0,          0]])
        Args:
            gate (Matrix): the matrix that represents a two-qubit gate
            qubit0 (int): id of the control qubit
            qubit1 (int): id of the target qubit
        """
        unitaty_add = self.enlarge_two_opt_sympy(gate, qubit0, qubit1, self._number_of_qubits)
        self._unitary_state = unitaty_add*self._unitary_state

    def run(self, q_job):
        """Run q_job

        Args:
            q_job (QuantumJob): job to run
        Returns:
            Result: Result is a class including the information to be returned to users.
                   Specifically, result_list in the return looks is important and it like this:
                   [
                     {'data': {'unitary':
                                       array([[sqrt(2)/2, sqrt(2)/2, 0, 0],
                                              [0, 0, sqrt(2)/2, -sqrt(2)/2],
                                              [0, 0, sqrt(2)/2, sqrt(2)/2],
                                              [sqrt(2)/2, -sqrt(2)/2, 0, 0]], dtype=object)},
                     'status': 'DONE'}
                    ]

        """
        # Generating a string id for the job
        job_id = str(uuid.uuid4())
        qobj = q_job.qobj
        result_list = []
        for circuit in qobj['circuits']:
            result_list.append(self.run_circuit(circuit))
        return Result({'job_id': job_id, 'result': result_list, 'status': 'COMPLETED'}, qobj)

    def run_circuit(self, circuit):
        """Run a circuit and return the results.
        Args:
            circuit (dict): JSON that describes the circuit
        Returns:
            dict: A dictionary of results which looks something like::
                {'data': {'unitary': array([[sqrt(2)/2, sqrt(2)/2, 0, 0],
                                            [0, 0, sqrt(2)/2, -sqrt(2)/2],
                                             [0, 0, sqrt(2)/2, sqrt(2)/2],
                                             [sqrt(2)/2, -sqrt(2)/2, 0, 0]], dtype=object)
                           },
                'status': 'DONE'}
        """
        ccircuit = circuit['compiled_circuit']
        self._number_of_qubits = ccircuit['header']['number_of_qubits']
        result = {}
        result['data'] = {}
        self._unitary_state = eye(2 ** self._number_of_qubits)
        for operation in ccircuit['operations']:
            if operation['name'] in ['U', 'u1', 'u2', 'u3']:
                if 'params' in operation:
                    params = operation['params']
                else:
                    params = None
                qubit = operation['qubits'][0]
                gate = SympyUnitarySimulator.compute_ugate_matrix_wrap(params)
                self._add_unitary_single(gate, qubit)
            elif operation['name'] in ['id']:
                logger.info('Warning have dropped identity gate from sympy-based unitary '
                            'simulator')
            elif operation['name'] in ['CX', 'cx']:
                qubit0 = operation['qubits'][0]
                qubit1 = operation['qubits'][1]
                gate = Matrix([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
                self._add_unitary_two(gate, qubit0, qubit1)
            elif operation['name'] == 'measure':
                logger.info('Warning have dropped measure from sympy-based unitary '
                            'simulator')
            elif operation['name'] == 'reset':
                logger.info('Warning have dropped reset from sympy-based unitary '
                            'simulator')
            elif operation['name'] == 'barrier':
                logger.info('Warning have dropped barrier from sympy-based unitary '
                            'simulator')
            else:
                result['status'] = 'ERROR'
                return result
        result['data']['unitary'] = np.array(self._unitary_state)
        result['status'] = 'DONE'
        return result
