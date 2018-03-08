# -*- coding: utf-8 -*-
# pylint: disable=invalid-name, no-name-in-module

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

import uuid
import random
from collections import Counter
import numpy as np
from sympy.physics.quantum.qubit import Qubit
from sympy.physics.quantum.qapply import qapply
from sympy import sqrt
from sympy.physics.quantum.qubit import matrix_to_qubit
from sympy import re, im
from sympy.physics.quantum.gate import H, X, Y, Z, S, T, CNOT, IdentityGate, OneQubitGate, CGate
from sympy.physics.quantum.represent import represent
from sympy import Matrix, pi, E, I, cos, sin, N, exp, nsimplify
from qiskit._result import Result
from qiskit.backends._basebackend import BaseBackend
from ._simulatorerror import SimulatorError


class SympyQasmSimulator(BaseBackend):
    """Python implementation of a qasm simulator."""
    def __init__(self, configuration=None):
        """
        Args:
            configuration (dict): backend configuration
        """
        super(SympyQasmSimulator, self).__init__()
        if configuration is None:
            self._configuration = {
                'name': 'local_sympy_qasm_simulator',
                'url': 'https://github.com/IBM/qiskit-sdk-py',
                'simulator': True,
                'local': True,
                'description': 'A python sympy-based simulator for qasm files',
                'coupling_map': 'all-to-all',
                'basis_gates': 'u1,u2,u3,cx,id'
            }
        else:
            self._configuration = configuration
        self._classical_state = None
        self._number_of_qubits = None
        self._number_of_cbits = None
        self._quantum_state = None
        self._shots = 1

    @staticmethod
    def _conjugate_square(com):
        return im(com)**2 + re(com)**2

    def _add_qasm_decision(self, qubit):
        """Apply the decision of measurement/reset qubit gate.
        qubit is the qubit that is measured/reset
        """
        probability_zero = 0
        random_number = random.random()
        vector_form = represent(self._quantum_state)
        for ii in range(1 << self._number_of_qubits):
            if ii & (1 << qubit) == 0:
                probability_zero += SympyQasmSimulator._conjugate_square(vector_form[ii, 0])
        probability_zero_sym = nsimplify(probability_zero, tolerance=0.001)
        if random_number <= probability_zero:
            outcome = '0'
            norm = sqrt(probability_zero_sym)
        else:
            outcome = '1'
            norm = sqrt(1-probability_zero_sym)
        return (outcome, norm)

    def _add_qasm_measure(self, qubit, cbit):
        """Apply the measurement qubit gate.
        qubit is the qubit measured.
        cbit is the classical bit the measurement is assigned to.
        """
        outcome, norm = self._add_qasm_decision(qubit)
        matrix_form = represent(self._quantum_state)

        for ii in range(1 << self._number_of_qubits):
            # update quantum state
            if (ii >> qubit) & 1 == int(outcome):
                matrix_form[ii, 0] = matrix_form[ii, 0]/norm
            else:
                matrix_form[ii, 0] = 0

        self._quantum_state = matrix_to_qubit(matrix_form)
        # update classical state
        bit = 1 << cbit
        self._classical_state = (self._classical_state & (~bit)) | (int(outcome) << cbit)

    def _add_qasm_reset(self, qubit):
        """Apply the reset to the qubit.

        This is done by doing a measruement and if 0 do nothing and
        if 1 flip the qubit.

        qubit is the qubit that is reset.
        """

        outcome, norm = self._add_qasm_decision(qubit)
        matrix_form_quantum_state = represent(self._quantum_state)
        matrix_form_temp = matrix_form_quantum_state[:, :]

        # self._quantum_state.fill(0.0)
        matrix_form_quantum_state = matrix_form_quantum_state - matrix_form_quantum_state

        # measurement
        for ii in range(1 << self._number_of_qubits):
            if (ii >> qubit) & 1 == int(outcome):
                matrix_form_temp[ii, 0] = matrix_form_temp[ii, 0]/norm
            else:
                matrix_form_temp[ii, 0] = 0
        # reset
        if outcome == '1':
            for ii in range(1 << self._number_of_qubits):
                iip = (~ (1 << qubit)) & ii  # bit number qubit set to zero
                matrix_form_quantum_state[iip, 0] += matrix_form_temp[ii, 0]
        else:
            matrix_form_quantum_state = matrix_form_temp

        self._quantum_state = matrix_to_qubit(matrix_form_quantum_state)

    def run(self, q_job):
        """Run circuits in q_job"""
        # Generating a string id for the job
        job_id = str(uuid.uuid4())
        qobj = q_job.qobj
        result_list = []
        self._shots = qobj['config']['shots']
        if self._shots > 1:
            print("Warning: no need for multiple shots! set shot=1 automatically!")
        for circuit in qobj['circuits']:
            result_list.append(self.run_circuit(circuit))
        return Result({'job_id': job_id, 'result': result_list, 'status': 'COMPLETED'}, qobj)

    def compute_distribution(self):
        """compute the distribution and return it in the list form"""
        matrix_form = represent(self._quantum_state)
        shapeN = matrix_form.shape[0]
        list_form = [SympyQasmSimulator._conjugate_square(matrix_form[i, 0]) for i in range(shapeN)]
        return list_form

    def run_circuit(self, circuit):
        """Run a circuit and return object
        Args:
            circuit (dict): JSON circuit from qobj circuits list
        Returns:
            dict: A dictionary of results which looks something like::
                {
                "data":{
                    "time"  : xx.xxxxxxxx
                    },
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
            elif operation['name'] in ['id', 'u0']:
                pass
            elif operation['name'] in ['CX', 'cx']:
                qubit0 = operation['qubits'][0]
                qubit1 = operation['qubits'][1]
                opname = operation['name'].upper()
                opparas = operation['params']
                q0q1tuple = tuple([qubit0, qubit1])
                _sym_op = SympyQasmSimulator.get_sym_op(opname, q0q1tuple, opparas)
                self._quantum_state = qapply(_sym_op * self._quantum_state)
            # Check if measure
            elif operation['name'] == 'measure':  # ignore the measure
                pass
            elif operation['name'] == 'reset':  # supported modification of state
                qubit = operation['qubits'][0]
                self._add_qasm_reset(qubit)
            elif operation['name'] == 'barrier':
                pass
            else:
                backend = globals()['__configuration']['name']
                err_msg = '{0} encountered unrecognized operation "{1}"'
                raise SimulatorError(err_msg.format(backend, operation['name']))

        outcomes = []
        matrix_form = represent(self._quantum_state)
        shapeN = matrix_form.shape[0]
        list_form = [matrix_form[i, 0] for i in range(shapeN)]

        pdist = [SympyQasmSimulator._conjugate_square(matrix_form[i, 0]) for i in range(shapeN)]
        norm_pdist = [float(i)/sum(pdist) for i in pdist]

        for i in range(actual_shots):  # pylint: disable=unused-variable
            _classical_state_observed = np.random.choice(np.arange(0, shapeN), p=norm_pdist)
            outcomes.append(bin(_classical_state_observed)[2:].zfill(
                self._number_of_cbits))

        # Return the results
        counts = dict(Counter(outcomes))
        data = {'counts': self._format_result(counts, cl_reg_index, cl_reg_nbits)}

        data['quantum_state'] = np.asarray(list_form)  # consistent with other backends
        data['classical_state'] = self._classical_state
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
    def regulate(theta):
        """find the symbolic form that is closest to the numeric value"""
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

    class SDGGate(OneQubitGate):  # pylint: disable=too-many-ancestors, abstract-method
        """implements the SDG gate"""
        gate_name = 'SDG'

        def get_target_matrix(self, format='sympy'):  # pylint: disable=redefined-builtin
            return Matrix([[1, 0], [0, -I]])

    class TDGGate(OneQubitGate):  # pylint: disable=too-many-ancestors, abstract-method
        """implements the TDG gate"""
        gate_name = 'TDG'

        def get_target_matrix(self, format='sympy'):  # pylint: disable=redefined-builtin
            return Matrix([[1, 0], [0, exp(-I*pi/4)]])

    @staticmethod
    def compute_ugate_matrix(parafloatlist):
        """compute the ugate matrix
        Args:
            parafloatlist (list): parameters
        Returns:
            Matrix: the u gate matrix
        """
        theta = parafloatlist[0]
        phi = parafloatlist[1]
        lamb = parafloatlist[2]

        theta, theta_is_regular = SympyQasmSimulator.regulate(theta)
        phi, phi_is_regular = SympyQasmSimulator.regulate(phi)
        lamb, lamb_is_regular = SympyQasmSimulator.regulate(lamb)

        uMat = Matrix([[cos(theta/2), (-E**(I*lamb)) * sin(theta/2)],
                       [(E**(I*phi)) * sin(theta/2), (E**(I*(phi+lamb))) * cos(theta/2)]])

        if theta_is_regular and phi_is_regular and lamb_is_regular:
            uMatNumeric = uMat
        else:
            uMatNumeric = uMat.evalf()
        return uMatNumeric

    @staticmethod
    def get_sym_op(name, qid_tuple, params=None):
        """ return the sympy version for the gate
        Args:
            name (str): gate name
            qid_tuple (tuple): the ids of the qubits being operated on
            params (list): optional parameter lists, needed by the U gates.
        Returns:
            object: gate applied to the qubits
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
            the_gate = SympyQasmSimulator.SDGGate(*qid_tuple)
        elif name == 'T':
            the_gate = T(*qid_tuple)
        elif name == 'TDG':
            the_gate = SympyQasmSimulator.TDGGate(*qid_tuple)
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
        else:  # U gate or CU gate
            if name.startswith('U') or name.startswith('CU'):
                parafloatlist = params

                if len(parafloatlist) == 1:  # [theta=0, phi=0, lambda]
                    parafloatlist.insert(0, 0.0)
                    parafloatlist.insert(0, 0.0)
                elif len(parafloatlist) == 2:  # [theta=pi/2, phi, lambda]
                    parafloatlist.insert(0, pi/2)
                elif len(parafloatlist) == 3:  # [theta, phi, lambda]
                    pass
                else:
                    return NotImplemented

                uMat = SympyQasmSimulator.compute_ugate_matrix(parafloatlist)

                class UGatePeng(OneQubitGate):  # pylint: disable=too-many-ancestors,abstract-method
                    """implements the general U gate"""
                    gate_name = 'U'

                    def get_target_matrix(self, format='sympy'):  # pylint:disable=redefined-builtin
                        return uMat

                if name.startswith('U'):
                    return UGatePeng(*qid_tuple)
                elif name.startswith('CU'):  # additional treatment for CU1, CU2, CU3
                    return CGate(qid_tuple[0], UGatePeng(*qid_tuple[1:]))
            elif name == "MEASURE":
                return None
            else:
                raise Exception('Not supported')
