# -*- coding: utf-8 -*-
# pylint: disable=unused-import

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

"""Backend for the Project Q C++ simulator."""


import itertools
import operator
import random
import uuid
import logging
from collections import OrderedDict, Counter
import numpy as np
from qiskit._result import Result
from qiskit.backends._basebackend import BaseBackend
from qiskit.backends._simulatorerror import SimulatorError
try:
    from projectq.backends._sim._cppsim import Simulator as CppSim
except ImportError:
    CppSim = None
else:
    from projectq import MainEngine
    from projectq.backends import Simulator
    from projectq.ops import (H,
                              X,
                              Y,
                              Z,
                              S,
                              T,
                              Rx,
                              Ry,
                              Rz,
                              CX,
                              Toffoli,
                              Measure,
                              BasicGate,
                              BasicMathGate,
                              QubitOperator,
                              TimeEvolution,
                              All)
logger = logging.getLogger(__name__)


class ProjectQSimulator(BaseBackend):
    """Python interface to Project Q simulator"""

    def __init__(self, configuration=None):
        """
        Args:
            configuration (dict): backend configuration
        Raises:
             ImportError: if the Project Q simulator is not available.
        """
        super().__init__(configuration)
        if CppSim is None:
            logger.info('Project Q C++ simulator unavailable.')
            raise ImportError('Project Q C++ simulator unavailable.')
        if configuration is None:
            self._configuration = {
                'name': 'local_projectq_simulator',
                'url': 'https://projectq.ch',
                'simulator': True,
                'local': True,
                'description': 'ProjectQ C++ simulator',
                'coupling_map': 'all-to-all',
                'basis_gates': 'h,s,t,cx,id'
            }
        else:
            self._configuration = configuration

        # Define the attributes inside __init__.
        self._number_of_qubits = 0
        self._number_of_clbits = 0
        self._quantum_state = 0
        self._classical_state = 0
        self._seed = None
        self._shots = 0
        self._sim = None

    def run(self, q_job):
        """Run circuits in q_job"""
        # Generating a string id for the job
        job_id = str(uuid.uuid4())
        result_list = []
        qobj = q_job.qobj
        self._sim = Simulator(gate_fusion=True)
        if 'seed' in qobj['config']:
            self._seed = qobj['config']['seed']
            self._sim._simulator = CppSim(self._seed)
        else:
            self._seed = random.getrandbits(32)
        self._shots = qobj['config']['shots']
        for circuit in qobj['circuits']:
            result_list.append(self.run_circuit(circuit))
        return Result({'job_id': job_id, 'result': result_list,
                       'status': 'COMPLETED'},
                      qobj)

    def run_circuit(self, circuit):
        """Run a circuit and return a single Result.

        Args:
            circuit (dict): JSON circuit from qobj circuits list

        Returns:
            dict: A dictionary of results which looks something like::

                {
                "data":
                    {  #### DATA CAN BE A DIFFERENT DICTIONARY FOR EACH BACKEND ####
                    "counts": {'00000': XXXX, '00001': XXXXX},
                    "time"  : xx.xxxxxxxx
                    },
                "status": --status (string)--
                }
        Raises:
            SimulatorError: if an error occurred.
        """
        # pylint: disable=expression-not-assigned,pointless-statement
        ccircuit = circuit['compiled_circuit']
        self._number_of_qubits = ccircuit['header']['number_of_qubits']
        self._number_of_clbits = ccircuit['header']['number_of_clbits']
        self._quantum_state = 0
        self._classical_state = 0
        cl_reg_index = []  # starting bit index of classical register
        cl_reg_nbits = []  # number of bits in classical register
        clbit_index = 0
        qobj_quregs = OrderedDict(_get_register_specs(
            ccircuit['header']['qubit_labels']))
        eng = MainEngine(backend=self._sim)
        for cl_reg in ccircuit['header']['clbit_labels']:
            cl_reg_nbits.append(cl_reg[1])
            cl_reg_index.append(clbit_index)
            clbit_index += cl_reg[1]
        # let circuit seed override qobj default
        if 'config' in circuit:
            if 'seed' in circuit['config']:
                if circuit['config']['seed'] is not None:
                    self._sim._simulator = CppSim(circuit['config']['seed'])
        outcomes = []
        for _ in range(self._shots):
            self._quantum_state = np.zeros(1 << self._number_of_qubits,
                                           dtype=complex)
            self._quantum_state[0] = 1
            # initialize starting state
            self._classical_state = 0
            unmeasured_qubits = list(range(self._number_of_qubits))
            projq_qureg_dict = OrderedDict(((key, eng.allocate_qureg(size))
                                            for key, size in
                                            qobj_quregs.items()))
            qureg = [qubit for sublist in projq_qureg_dict.values()
                     for qubit in sublist]
            # Do each operation in this shot
            for operation in ccircuit['operations']:
                if 'conditional' in operation:
                    mask = int(operation['conditional']['mask'], 16)
                    if mask > 0:
                        value = self._classical_state & mask
                        while (mask & 0x1) == 0:
                            mask >>= 1
                            value >>= 1
                        if value != int(operation['conditional']['val'], 16):
                            continue
                # Check if single  gate
                if operation['name'] in ['U', 'u3']:
                    params = operation['params']
                    qubit = qureg[operation['qubits'][0]]
                    Rz(params[2]) | qubit
                    Ry(params[0]) | qubit
                    Rz(params[1]) | qubit
                elif operation['name'] in ['u1']:
                    params = operation['params']
                    qubit = qureg[operation['qubits'][0]]
                    Rz(params[0]) | qubit
                elif operation['name'] in ['u2']:
                    params = operation['params']
                    qubit = qureg[operation['qubits'][0]]
                    Rz(params[1] - np.pi/2) | qubit
                    Rx(np.pi/2) | qubit
                    Rz(params[0] + np.pi/2) | qubit
                elif operation['name'] == 't':
                    qubit = qureg[operation['qubits'][0]]
                    T | qubit
                elif operation['name'] == 'h':
                    qubit = qureg[operation['qubits'][0]]
                    H | qubit
                elif operation['name'] == 's':
                    qubit = qureg[operation['qubits'][0]]
                    S | qubit
                elif operation['name'] in ['CX', 'cx']:
                    qubit0 = qureg[operation['qubits'][0]]
                    qubit1 = qureg[operation['qubits'][1]]
                    CX | (qubit0, qubit1)
                elif operation['name'] in ['id', 'u0']:
                    pass
                # Check if measure
                elif operation['name'] == 'measure':
                    qubit_index = operation['qubits'][0]
                    qubit = qureg[qubit_index]
                    clbit = operation['clbits'][0]
                    Measure | qubit
                    bit = 1 << clbit
                    self._classical_state = (
                        self._classical_state & (~bit)) | (int(qubit)
                                                           << clbit)
                    # check whether we already measured this qubit
                    if operation['qubits'][0] in unmeasured_qubits:
                        unmeasured_qubits.remove(operation['qubits'][0])
                # Check if reset
                elif operation['name'] == 'reset':
                    qubit = operation['qubits'][0]
                    raise SimulatorError('Reset operation not yet implemented '
                                         'for ProjectQ C++ backend')
                elif operation['name'] == 'barrier':
                    pass
                else:
                    backend = self._configuration['name']
                    err_msg = '{0} encountered unrecognized operation "{1}"'
                    raise SimulatorError(err_msg.format(backend,
                                                        operation['name']))
            for ind in unmeasured_qubits:
                qubit = qureg[ind]
                Measure | qubit
            eng.flush()
            # Turn classical_state (int) into bit string
            state = format(self._classical_state, 'b')
            outcomes.append(state.zfill(self._number_of_clbits))
        # Return the results
        counts = dict(Counter(outcomes))
        data = {'counts': _format_result(
            counts, cl_reg_index, cl_reg_nbits)}
        if self._shots == 1:
            data['quantum_state'] = self._quantum_state
            data['classical_state'] = self._classical_state
        return {'data': data, 'status': 'DONE'}


def _get_register_specs(bit_labels):
    """
    Get the number and size of unique registers from bit_labels list with an
    iterator of register_name:size pairs.

    Args:
        bit_labels (list): this list is of the form::

            [['reg1', 0], ['reg1', 1], ['reg2', 0]]

            which indicates a register named "reg1" of size 2
            and a register named "reg2" of size 1. This is the
            format of classic and quantum bit labels in qobj
            header.
    Yields:
        tuple: pairs of (register_name, size)
    """
    iterator = itertools.groupby(bit_labels, operator.itemgetter(0))
    for register_name, sub_it in iterator:
        yield register_name, max(ind[1] for ind in sub_it) + 1


def _format_result(counts, cl_reg_index, cl_reg_nbits):
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
