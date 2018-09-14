# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""Contains a (slow) python simulator.

It simulates a qasm quantum circuit that has been compiled to run on the
simulator. It is exponential in the number of qubits.

We advise using the c++ simulator or online simulator for larger size systems.

The input is a qobj dictionary

and the output is a Results object

    results['data']["counts"] where this is dict {"0000" : 454}

The simulator is run using

.. code-block:: python

    QasmSimulatorPy(compiled_circuit,shots,seed).run().

.. code-block:: guess

       compiled_circuit =
       {
        "header": {
        "number_of_qubits": 2, // int
        "number_of_clbits": 2, // int
        "qubit_labels": [["q", 0], ["v", 0]], // list[list[string, int]]
        "clbit_labels": [["c", 2]], // list[list[string, int]]
        }
        "operations": // list[map]
           [
               {
                   "name": , // required -- string
                   "params": , // optional -- list[double]
                   "qubits": , // required -- list[int]
                   "clbits": , // optional -- list[int]
                   "conditional":  // optional -- map
                       {
                           "type": , // string
                           "mask": , // hex string
                           "val":  , // bhex string
                       }
               },
           ]
       }

.. code-block:: python

       result =
               {
               'data': {
                        'statevector': array([ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j]),
                        'classical_state': 0
                        'counts': {'0000': 1}
                        'snapshots': { '0': {'statevector': array([1.+0.j,  0.+0.j,
                                                                     0.+0.j,  0.+0.j])}}
                        }
                   }
               'time_taken': 0.002
               'status': 'DONE'
               }

"""
import random
import uuid
import time
import logging
from collections import Counter

import numpy as np

from qiskit.result._utils import copy_qasm_from_qobj_into_result, result_from_old_style_dict
from qiskit.backends import BaseBackend
from qiskit.backends.local.localjob import LocalJob
from ._simulatorerror import SimulatorError
from ._simulatortools import single_gate_matrix
logger = logging.getLogger(__name__)


class QasmSimulatorPy(BaseBackend):
    """Python implementation of a qasm simulator."""

    DEFAULT_CONFIGURATION = {
        'name': 'local_qasm_simulator_py',
        'url': 'https://github.com/QISKit/qiskit-terra',
        'simulator': True,
        'local': True,
        'description': 'A python simulator for qasm files',
        'coupling_map': 'all-to-all',
        'basis_gates': 'u1,u2,u3,cx,id,snapshot'
    }

    def __init__(self, configuration=None):
        """
        Args:
            configuration (dict): backend configuration
        """
        super().__init__(configuration or self.DEFAULT_CONFIGURATION.copy())

        self._local_random = random.Random()

        # Define attributes in __init__.
        self._classical_state = 0
        self._statevector = 0
        self._snapshots = {}
        self._number_of_cbits = 0
        self._number_of_qubits = 0
        self._shots = 0
        self._qobj_config = None

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
        assert i1 != i2

        if i1 > i2:
            # insert as (i1-1)th bit, will be shifted left 1 by next line
            retval = QasmSimulatorPy._index1(b1, i1-1, k)
            retval = QasmSimulatorPy._index1(b2, i2, retval)
        else:  # i2>i1
            # insert as (i2-1)th bit, will be shifted left 1 by next line
            retval = QasmSimulatorPy._index1(b2, i2-1, k)
            retval = QasmSimulatorPy._index1(b1, i1, retval)
        return retval

    def _add_qasm_single(self, gate, qubit):
        """Apply an arbitary 1-qubit operator to a qubit.

        Gate is the single qubit applied.
        qubit is the qubit the gate is applied to.
        """
        psi = self._statevector
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
        psi = self._statevector
        for k in range(0, 1 << (self._number_of_qubits - 2)):
            # first bit is control, second is target
            ind1 = self._index2(1, q0, 0, q1, k)
            # swap target if control is 1
            ind3 = self._index2(1, q0, 1, q1, k)
            cache0 = psi[ind1]
            cache1 = psi[ind3]
            psi[ind3] = cache0
            psi[ind1] = cache1

    def _add_qasm_decision(self, qubit):
        """Apply the decision of measurement/reset qubit gate.

        qubit is the qubit that is measured/reset
        """
        probability_zero = 0
        random_number = self._local_random.random()
        for ii in range(1 << self._number_of_qubits):
            if ii & (1 << qubit) == 0:
                probability_zero += np.abs(self._statevector[ii])**2
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
                self._statevector[ii] = self._statevector[ii]/norm
            else:
                self._statevector[ii] = 0
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
        temp = np.copy(self._statevector)
        self._statevector.fill(0.0)
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
                self._statevector[iip] += temp[ii]
        else:
            self._statevector = temp

    def _add_qasm_snapshot(self, slot):
        """Snapshot instruction to record simulator's internal representation
        of quantum statevector.

        slot is an integer indicating a snapshot slot number.
        """
        self._snapshots.setdefault(str(int(slot)),
                                   {}).setdefault("statevector",
                                                  []).append(np.copy(self._statevector))

    def run(self, qobj):
        """Run qobj asynchronously.

        Args:
            qobj (dict): job description

        Returns:
            LocalJob: derived from BaseJob
        """
        local_job = LocalJob(self._run_job, qobj)
        local_job.submit()
        return local_job

    def _run_job(self, qobj):
        """Run circuits in qobj"""
        self._validate(qobj)
        result_list = []
        self._shots = qobj.config.shots
        self._qobj_config = qobj.config
        start = time.time()

        for circuit in qobj.experiments:
            result_list.append(self.run_circuit(circuit))
        end = time.time()
        job_id = str(uuid.uuid4())
        result = {'backend': self._configuration['name'],
                  'id': qobj.qobj_id,
                  'job_id': job_id,
                  'result': result_list,
                  'status': 'COMPLETED',
                  'success': True,
                  'time_taken': (end - start)}

        copy_qasm_from_qobj_into_result(qobj, result)

        return result_from_old_style_dict(
            result, [circuit.header.name for circuit in qobj.experiments])

    def run_circuit(self, circuit):
        """Run a circuit and return a single Result.

        Args:
            circuit (QobjExperiment): experiment from qobj experiments list

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
        self._number_of_qubits = circuit.header.number_of_qubits
        self._number_of_cbits = circuit.header.number_of_clbits
        self._statevector = 0
        self._classical_state = 0
        self._snapshots = {}
        cl_reg_index = []  # starting bit index of classical register
        cl_reg_nbits = []  # number of bits in classical register
        cbit_index = 0
        for cl_reg in circuit.header.clbit_labels:
            cl_reg_nbits.append(cl_reg[1])
            cl_reg_index.append(cbit_index)
            cbit_index += cl_reg[1]

        # Get the seed looking in circuit, qobj, and then random.
        seed = getattr(circuit.config, 'seed',
                       getattr(self._qobj_config, 'seed',
                               random.getrandbits(32)))
        self._local_random.seed(seed)
        outcomes = []

        start = time.time()
        for _ in range(self._shots):
            self._statevector = np.zeros(1 << self._number_of_qubits,
                                         dtype=complex)
            self._statevector[0] = 1
            self._classical_state = 0
            for operation in circuit.instructions:
                if getattr(operation, 'conditional', None):
                    mask = int(operation.conditional.mask, 16)
                    if mask > 0:
                        value = self._classical_state & mask
                        while (mask & 0x1) == 0:
                            mask >>= 1
                            value >>= 1
                        if value != int(operation.conditional.val, 16):
                            continue
                # Check if single  gate
                if operation.name in ('U', 'u1', 'u2', 'u3'):
                    params = getattr(operation, 'params', None)
                    qubit = operation.qubits[0]
                    gate = single_gate_matrix(operation.name, params)
                    self._add_qasm_single(gate, qubit)
                # Check if CX gate
                elif operation.name in ('id', 'u0'):
                    pass
                elif operation.name in ('CX', 'cx'):
                    qubit0 = operation.qubits[0]
                    qubit1 = operation.qubits[1]
                    self._add_qasm_cx(qubit0, qubit1)
                # Check if measure
                elif operation.name == 'measure':
                    qubit = operation.qubits[0]
                    cbit = operation.clbits[0]
                    self._add_qasm_measure(qubit, cbit)
                # Check if reset
                elif operation.name == 'reset':
                    qubit = operation.qubits[0]
                    self._add_qasm_reset(qubit)
                # Check if barrier
                elif operation.name == 'barrier':
                    pass
                # Check if snapshot command
                elif operation.name == 'snapshot':
                    params = operation.params
                    self._add_qasm_snapshot(params[0])
                else:
                    backend = self._configuration['name']
                    err_msg = '{0} encountered unrecognized operation "{1}"'
                    raise SimulatorError(err_msg.format(backend,
                                                        operation.name))
            # Turn classical_state (int) into bit string
            outcomes.append(bin(self._classical_state)[2:].zfill(
                self._number_of_cbits))
        # Return the results
        counts = dict(Counter(outcomes))
        data = {
            'counts': self._format_result(counts, cl_reg_index, cl_reg_nbits),
            'snapshots': self._snapshots
        }
        if self._shots == 1:
            # TODO: deprecated -- remove in v0.6
            data['statevector'] = self._statevector
            data['quantum_state'] = self._statevector
            data['classical_state'] = self._classical_state
        end = time.time()
        return {'name': circuit.header.name,
                'seed': seed,
                'shots': self._shots,
                'data': data,
                'status': 'DONE',
                'success': True,
                'time_taken': (end-start)}

    def _validate(self, qobj):
        for experiment in qobj.experiments:
            if 'measure' not in [op.name for
                                 op in experiment.instructions]:
                logger.warning("no measurements in circuit '%s', "
                               "classical register will remain all zeros.",
                               experiment.header.name)

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
            if cl_reg_nbits:
                new_key = [key[-cl_reg_nbits[0]:]]
                for index, nbits in zip(cl_reg_index[1:],
                                        cl_reg_nbits[1:]):
                    new_key.insert(0, key[-(index+nbits):-index])
                fcounts[' '.join(new_key)] = value
        return fcounts
