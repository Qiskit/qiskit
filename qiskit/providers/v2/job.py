# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from abc import ABC
from abc import abstractmethod
from collections import OrderedDict
import warnings

import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.pulse.schedule import Schedule
from qiskit.exceptions import QiskitError
from qiskit.result.counts import Counts
from qiskit.result.result import Result
from qiskit.quantum_info.states.statevector import Statevector


class Job(ABC):
    """A base class for representing an async job on a backend.

    This provides the base structure required for building an async or
    sync job handler. The underlying concept around this is that this
    object gets returned by :meth:`qiskit.providers.v2.Backend.run` and
    depending on the nature of the backend will either be the record of
    the execution with results, or in the case of async backends will
    be a handle to the record of the async execution that will populate
    the result data after the execution is complete. The ``result_data``
    instance attribute can either be the data from a single execution
    (depending on the backend this would be a
    :class:`~qiskit.circuit.QuantumCircuit`, :class:`~qiskit.result.Counts`,
    :class:`~qiskit.quantum_info.Statevector`, etc. object), or in the case
    there were multiple inputs an ``OrderedDict`` class where the key is the
    name of the input circuit or schedule.
    """

    job_id = None
    metadata = None

    def __init__(self, job_id, backend, time_taken=None):
        """Initialize a new job object

        Args:
            job_id (str): A unique identifier for the job
            backend (qiskit.providers.v2.BaseBackend): The backend the job is
                being run on
            time_taken (float): Te duration of the job in seconds, should only
                be set during init for sync jobs.
        """

        self.result_data = None
        self.job_id = job_id
        self.backend = backend
        self.time_taken = None

    def _result_conversion(self, result, name=None):
        if isinstance(result, Counts):
            if 'seed_simulator' in result.metadata:
                seed_simulator = result.metadata.pop(
                    'seed_simulator')
            else:
                seed_simulator = None
            header = result.metadata
            header['name'] = result.name
            result_dict = {
                'shots': result.shots,
                'data': result.hex_raw,
                'success': True,
                'time_taken': result.time_taken,
                'header': header,
            }
            if seed_simulator:
                result_dict['seed_simulator'] = seed_simulator
                result_dict['seed'] = seed_simulator
        elif isinstance(result, Statevector):
            result_dict = {
                'data': result.data,
                'status': 'DONE',
                'success': True,
            }
        elif isinstance(result, np.ndarray):
            if name:
                header = {'name': name}
            else:
                header = {}
            result_dict = {
                'data': {'unitary': result},
                'status': 'DONE',
                'success': True,
                'shots': 1,
                'header': header
                }
        else:
            raise Exception
        return result_dict

    def result(self):
        result_list = {}
        self.wait_for_final_state()
        warnings.warn("The result method is deprecated instead access the "
                      "result_data attribute to access the result from the "
                      "job", DeprecationWarning, stacklevel=2)
        result_dict = {
            'backend_name': self.backend.name,
            'qobj_id': '',
            'backend_version': '',
            'success': True,
            'job_id': self.job_id,
        }
        if isinstance(self.result_data, OrderedDict):
            result_list = []
            for result in self.result_data:
                result_list.append(
                    self._result_conversion(self.result_data[result], result))
        elif isinstance(self.result_data, Counts, Statevector, np.ndarray):
            result_list = [self._result_conversion(self.result_data)]
        else:
            raise TypeError(
                "Result for job %s is not a circuit result and a backwards "
                "compat Result object can't be constructed for "
                "it" % self.job_id)
        result_dict['results'] = result_list
        return Result.from_dict(result_dict)

    @abstractmethod
    def status(self):
        pass

    def cancel(self):
        raise NotImplementedError

    @abstractmethod
    def wait_for_final_state():
        pass

    def _get_experiment(self, key=None):
        # Automatically return the first result if no key was provided.
        if key is None:
            if isinstance(self.result_data,
                          OrderedDict) and len(self.result_data) != 1:
                raise QiskitError(
                    'You have to select a circuit or schedule when there is more than '
                    'one available')
            key = 0
        else:
            if not isinstance(self.result_data, OrderedDict):
                raise QiskitError("You can't specify a key if there is only "
                                  "one result")

        # Key is a QuantumCircuit/Schedule or str: retrieve result by name.
        if isinstance(key, (QuantumCircuit, Schedule)):
            key = key.name
        # Key is an integer: return result by index.
        if isinstance(key, int):
            return list(self.result_data.values())[key]
        elif isinstance(key, str):
            return self.result_data[key]
        else:
            raise TypeError('Invalid key type %s' % type(key))

    def get_memory(self):
        raise NotImplementedError

    def get_counts(self, experiment=None):
        self.wait_for_final_state()
        if isinstance(self.result_data, Counts):
            return self.result_data
        elif isinstance(self.result_data, OrderedDict):
            exp_result = self._get_experiment(experiment)
            if isinstance(exp_result, Counts):
                return exp_result
            else:
                raise TypeError(
                    "Result for job %s is not a Counts object" % self.job_id)
        else:
            raise TypeError(
                "Result for job %s is not a Counts object" % self.job_id)

    def get_statevector(self, experiment=None):
        self.wait_for_final_state()
        if isinstance(self.result_data, Statevector):
            return self.result_data
        elif isinstance(self.result_data, OrderedDict):
            exp_result = self._get_experiment(experiment)
            if isinstance(exp_result, Statevector):
                return exp_result
            else:
                raise TypeError(
                    "Result for job %s is not a Statevector "
                    "object" % self.job_id)
        else:
            raise TypeError(
                "Result for job %s is not a Statevector object" % self.job_id)

    def get_unitary(self, experiment=None):
        self.wait_for_final_state()
        if isinstance(self.result_data, np.array):
            return self.result_data
        elif isinstance(self.result_data, OrderedDict):
            exp_result = self._get_experiment(experiment)
            if isinstance(exp_result, np.ndarray):
                return exp_result
            else:
                raise TypeError(
                    "Result for job %s is not a unitary array" % self.job_id)
        else:
            raise TypeError(
                "Result for job %s is not a Statevector object" % self.job_id)
