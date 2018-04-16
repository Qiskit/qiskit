# -*- coding: utf-8 -*-
# pylint: disable=abstract-method,too-many-ancestors

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
"""
Contains a (slow) python sympy-based simulator.

It produces the state vector in symbolic form.
In particular, it simulates the quantum computation with the sympy APIs,
which preserve the symbolic form of numbers, e.g., sqrt(2), e^{i*pi/2}.

How to use this simulator:
see examples/python/use_sympy_simulators.py

Example output:
final quantum amplitude vector: [sqrt(2)/2 0 0 sqrt(2)/2]

Advantages:
1. The tool obviates the manual calculation with a pen and paper, enabling
 quick adjustment of your prototype code.
2. The tool leverages sympy's symbolic computational power to keep the most
leverages sympy's simplification engine to simplify the expressions as much as possible.
3. The tool supports u gates, including u1, u2, u3, cu1, cu2, cu3.

Analysis of results and limitations:
1. It can simplify expressions, including complex ones such as sqrt(2)*I*exp(-I*pi/4)/4.
2. It may miss some simplification opportunities.
For instance, the amplitude
"0.245196320100808*sqrt(2)*exp(-I*pi/4) - 0.048772580504032*sqrt(2)*I*exp(-I*pi/4)"
can be further simplified.
3. It may produce results that are hard to interpret.
4. Memory error may occur if there are many qubits in the system.
This is due to the limit of classical computers and show the advantage of the quantum hardware.

Warning: it is slow.
Warning: this simulator computes the final amplitude vector precisely within a single shot.
Therefore we do not need multiple shots.
If you specify multiple shots, it will automatically set shots=1.
"""

import logging
import random
import uuid
from collections import Counter

import numpy as np
from sympy import Matrix, pi, I, exp
from sympy import re, im
from sympy.physics.quantum.gate import H, X, Y, Z, S, T, CNOT, IdentityGate, OneQubitGate, CGate
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qubit import Qubit
from sympy.physics.quantum.represent import represent

from qiskit._result import Result
from qiskit.backends.basebackend import BaseBackend
from qiskit.backends.local._simulatorerror import SimulatorError
from qiskit.backends.local._simulatortools import compute_ugate_matrix

logger = logging.getLogger(__name__)


class SDGGate(OneQubitGate):
    """implements the SDG gate"""
    gate_name = 'SDG'

    def get_target_matrix(self, format='sympy'):
        """Return the Matrix that corresponds to the gate.

        Returns:
            Matrix: the matrix that corresponds to the gate.
                    Matrix is a type from sympy.
                    Each entry in it can be in the symbolic form.
        """
        # pylint: disable=redefined-builtin
        return Matrix([[1, 0], [0, -I]])


class TDGGate(OneQubitGate):
    """implements the TDG gate"""
    gate_name = 'TDG'

    def get_target_matrix(self, format='sympy'):
        """Return the Matrix that corresponds to the gate.

        Returns:
            Matrix: the matrix that corresponds to the gate
        """
        # pylint: disable=redefined-builtin
        return Matrix([[1, 0], [0, exp(-I*pi/4)]])


class UGateGeneric(OneQubitGate):
    """implements the general U gate"""
    _u_mat = None
    gate_name = 'U'

    def set_target_matrix(self, u_matrix):
        """this API sets the raw matrix that corresponds to the U gate
            the client should use this API whenever she creates a UGateGeneric object!
            Args:
                u_matrix (Matrix): set the matrix that corresponds to the gate
        """
        self._u_mat = u_matrix

    def get_target_matrix(self, format='sympy'):
        """return the Matrix that corresponds to the gate
        Returns:
            Matrix: the matrix that corresponds to the gate
        """
        # pylint: disable=redefined-builtin
        return self._u_mat


class SympyQasmSimulator(BaseBackend):
    """Python implementation of a qasm simulator."""
    DEFAULT_CONFIGURATION = {
        'name': 'local_sympy_qasm_simulator',
        'url': 'https://github.com/QISKit/qiskit-sdk-py',
        'simulator': True,
        'local': True,
        'description': 'A python sympy-based simulator for qasm files',
        'coupling_map': 'all-to-all',
        'basis_gates': 'u1,u2,u3,cx,id'
    }

    def __init__(self, configuration=None):
        """
        Args:
            configuration (dict): backend configuration
        """
        super().__init__(configuration or self.DEFAULT_CONFIGURATION.copy())

        self._classical_state = None
        self._number_of_qubits = None
        self._number_of_cbits = None
        self._quantum_state = None
        self._shots = 1

    @staticmethod
    def _conjugate_square(com):
        """simpler helper for returning com*conjugate(com), where com is a complex number
            Args:
                com (object): a complex number
            Returns:
                object: com*conjugate(com)
        """
        return im(com)**2 + re(com)**2

    def run(self, q_job):
        """Run circuits in q_job and return the result
            Args:
                q_job (QuantumJob): all the information necessary
                    (e.g., circuit, backend and resources) for running a circuit
            Returns:
                Result: Result is a class including the information to be returned to users.
                    Specifically, result_list in the return contains the essential information,
                    which looks like this::
                        [{'data':
                        {
                          'counts': {'00': 1},
                          'quantum_state': array([sqrt(2)/2, 0, 0, sqrt(2)/2], dtype=object),
                          'classical_state': 0
                        },
                        'status': 'DONE'
                        }]
        """
        job_id = str(uuid.uuid4())
        qobj = q_job.qobj
        result_list = []
        self._shots = qobj['config']['shots']
        if self._shots > 1:
            logger.info("No need for multiple shots. A single execution will be performed.")
        for circuit in qobj['circuits']:
            result_list.append(self.run_circuit(circuit))
        return Result({'job_id': job_id, 'result': result_list, 'status': 'COMPLETED'}, qobj)

    def run_circuit(self, circuit):
        """Run a circuit and return object
        Args:
            circuit (dict): JSON that describes the circuit
        Returns:
            dict: A dictionary of results which looks something like::
                {
                "data":{
                        'classical_state': 0,
                        'counts': {'11': 1},
                        'quantum_state': array([sqrt(2)/2, 0, 0, sqrt(2)/2], dtype=object)},
                "status": --status (string)--
                }
        Raises:
            SimulatorError: if an error occurred.
        """
        ccircuit = circuit['compiled_circuit']
        self._number_of_qubits = ccircuit['header']['number_of_qubits']
        self._number_of_cbits = ccircuit['header']['number_of_clbits']
        self._quantum_state = 0
        self._classical_state = 0
        cl_reg_index = []  # starting bit index of classical register
        cl_reg_nbits = []  # number of bits in classical register
        cbit_index = 0
        for cl_reg in ccircuit['header']['clbit_labels']:
            cl_reg_nbits.append(cl_reg[1])
            cl_reg_index.append(cbit_index)
            cbit_index += cl_reg[1]
        if circuit['config']['seed'] is None:
            random.seed(random.getrandbits(32))
        else:
            random.seed(circuit['config']['seed'])

        actual_shots = self._shots
        self._quantum_state = Qubit(*tuple([0]*self._number_of_qubits))
        self._classical_state = 0
        # Do each operation in this shot
        for operation in ccircuit['operations']:
            if 'conditional' in operation:  # not related to sympy
                mask = int(operation['conditional']['mask'], 16)
                if mask > 0:
                    value = self._classical_state & mask
                    while (mask & 0x1) == 0:
                        mask >>= 1
                        value >>= 1
                    if value != int(operation['conditional']['val'], 16):
                        continue
            if operation['name'] in ['U', 'u1', 'u2', 'u3']:
                qubit = operation['qubits'][0]
                opname = operation['name'].upper()
                opparas = operation['params']
                _sym_op = SympyQasmSimulator.get_sym_op(opname, tuple([qubit]), opparas)
                _applied_quantum_state = _sym_op * self._quantum_state
                self._quantum_state = qapply(_applied_quantum_state)
            # Check if CX gate
            elif operation['name'] in ['id']:
                logger.info('Identity gate is ignored by sympy-based qasm simulator.')
            elif operation['name'] in ['CX', 'cx']:
                qubit0 = operation['qubits'][0]
                qubit1 = operation['qubits'][1]
                opname = operation['name'].upper()
                if 'params' in operation:
                    opparas = operation['params']
                else:
                    opparas = None
                q0q1tuple = tuple([qubit0, qubit1])
                _sym_op = SympyQasmSimulator.get_sym_op(opname, q0q1tuple, opparas)
                self._quantum_state = qapply(_sym_op * self._quantum_state)
            # Check if measure
            elif operation['name'] == 'measure':
                logger.info('The statement measure is ignored by sympy-based qasm simulator.')
            elif operation['name'] == 'reset':
                logger.info('The statement reset is ignored by sympy-based qasm simulator.')
            elif operation['name'] == 'barrier':
                logger.info('The statement barrier is ignored by sympy-based qasm simulator.')
            else:
                backend = globals()['__configuration']['name']
                err_msg = '{0} encountered unrecognized operation "{1}"'
                raise SimulatorError(err_msg.format(backend, operation['name']))

        outcomes = []
        matrix_form = represent(self._quantum_state)
        shape_n = matrix_form.shape[0]
        list_form = [matrix_form[i, 0] for i in range(shape_n)]

        pdist = [SympyQasmSimulator._conjugate_square(matrix_form[i, 0]) for i in range(shape_n)]
        norm_pdist = [float(i)/sum(pdist) for i in pdist]

        for _ in range(actual_shots):
            _classical_state_observed = np.random.choice(np.arange(0, shape_n), p=norm_pdist)
            outcomes.append(bin(_classical_state_observed)[2:].zfill(
                self._number_of_cbits))

        # Return the results
        counts = dict(Counter(outcomes))
        # data['quantum_state']: consistent with other backends.
        data = {
            'counts': self._format_result(counts, cl_reg_index, cl_reg_nbits),
            'quantum_state': np.asarray(list_form),
            'classical_state': self._classical_state
        }

        return {'data': data, 'status': 'DONE'}

    def _format_result(self, counts, cl_reg_index, cl_reg_nbits):
        """Format the result bit string.
        This formats the result bit strings such that spaces are inserted
        at register divisions.

        Args:
            counts (dict): dictionary of counts e.g. {'1111': 1000, '0000':5}
            cl_reg_index (list): starting bit index of classical register
            cl_reg_nbits (list): total amount of bits in classical register
        Returns:
            dict: spaces inserted into dictionary keys at register boundaries.
        """
        fcounts = {}
        for key, value in counts.items():
            new_key = [key[-cl_reg_nbits[0]:]]
            for index, nbits in zip(cl_reg_index[1:],
                                    cl_reg_nbits[1:]):
                new_key.insert(0, key[-(index+nbits):-index])
            fcounts[' '.join(new_key)] = value
        return fcounts

    @staticmethod
    def get_sym_op(name, qid_tuple, params=None):
        """ return the sympy version for the gate
        Args:
            name (str): gate name
            qid_tuple (tuple): the ids of the qubits being operated on
            params (list): optional parameter lists, which may be needed by the U gates.
        Returns:
            object: (the sympy representation of) the gate being applied to the qubits
        Raises:
            Exception: if an unsupported operation is seen
        """
        the_gate = None
        if name == 'ID':
            the_gate = IdentityGate(*qid_tuple)  # de-tuple means unpacking
        elif name == 'X':
            the_gate = X(*qid_tuple)
        elif name == 'Y':
            the_gate = Y(*qid_tuple)
        elif name == 'Z':
            the_gate = Z(*qid_tuple)
        elif name == 'H':
            the_gate = H(*qid_tuple)
        elif name == 'S':
            the_gate = S(*qid_tuple)
        elif name == 'SDG':
            the_gate = SDGGate(*qid_tuple)
        elif name == 'T':
            the_gate = T(*qid_tuple)
        elif name == 'TDG':
            the_gate = TDGGate(*qid_tuple)
        elif name == 'CX' or name == 'CNOT':
            the_gate = CNOT(*qid_tuple)
        elif name == 'CY':
            the_gate = CGate(qid_tuple[0], Y(qid_tuple[1]))  # qid_tuple: control target
        elif name == 'CZ':
            the_gate = CGate(qid_tuple[0], Z(qid_tuple[1]))  # qid_tuple: control target
        elif name == 'CCX' or name == 'CCNOT' or name == 'TOFFOLI':
            the_gate = CGate((qid_tuple[0], qid_tuple[1]), X(qid_tuple[2]))

        if the_gate is not None:
            return the_gate

        # U gate, CU gate or measure gate handled below
        if name.startswith('U') or name.startswith('CU'):
            parameters = params

            if len(parameters) == 1:  # [theta=0, phi=0, lambda]
                parameters.insert(0, 0.0)
                parameters.insert(0, 0.0)
            elif len(parameters) == 2:  # [theta=pi/2, phi, lambda]
                parameters.insert(0, pi/2)
            elif len(parameters) == 3:  # [theta, phi, lambda]
                pass
            else:
                raise Exception('U gate must carry 1, 2 or 3 parameters!')

            if name.startswith('U'):
                ugate = UGateGeneric(*qid_tuple)
                u_mat = compute_ugate_matrix(parameters)
                ugate.set_target_matrix(u_matrix=u_mat)
                return ugate

            elif name.startswith('CU'):  # additional treatment for CU1, CU2, CU3
                ugate = UGateGeneric(*qid_tuple)
                u_mat = compute_ugate_matrix(parameters)
                ugate.set_target_matrix(u_matrix=u_mat)
                return CGate(qid_tuple[0], ugate)
        elif name == "MEASURE":
            return None
        # if the control flow comes here,  alarm!
        raise Exception('Not supported')
