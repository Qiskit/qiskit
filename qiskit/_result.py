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

"""Module for working with Results."""

import copy
import numpy
from qiskit import QISKitError
from qiskit import QuantumCircuit


class Result(object):
    """ Result Class.

    Class internal properties.

    Methods to process the quantum program after it has been run

    Internal::

        qobj =  { -- the quantum object that was complied --}
        result = {
            "job_id": --job-id (string),
                      #This string links the result with the job that computes it,
                      #it should be issued by the backend it is run on.
            "status": --status (string),
            "result":
                [
                    {
                    "data":
                        {  #### DATA CAN BE A DIFFERENT DICTIONARY FOR EACH BACKEND ####
                        "counts": {'00000': XXXX, '00001': XXXXX},
                        "time"  : xx.xxxxxxxx
                        },
                    "status": --status (string)--
                    },
                    ...
                ]
            }
    """

    def __init__(self, qobj_result, qobj):
        self._result = qobj_result
        self._qobj = qobj

    def __str__(self):
        """Get the status of the run.

        Returns:
            string: the status of the results.
        """
        return self._result['status']

    def __getitem__(self, i):
        return self._result['result'][i]

    def __len__(self):
        return len(self._result['result'])

    def __iadd__(self, other):
        """Append a Result object to current Result object.

        Arg:
            other (Result): a Result object to append.
        Returns:
            Result: The current object with appended results.
        Raises:
            QISKitError: if the Results cannot be combined.
        """
        if self._qobj['config'] == other._qobj['config']:
            if isinstance(self._qobj['id'], str):
                self._qobj['id'] = [self._qobj['id']]
            self._qobj['id'].append(other._qobj['id'])
            self._qobj['circuits'] += other._qobj['circuits']
            self._result['result'] += other._result['result']
            return self
        else:
            raise QISKitError('Result objects have different configs and cannot be combined.')

    def __add__(self, other):
        """Combine Result objects.

        Note that the qobj id of the returned result will be the same as the
        first result.

        Arg:
            other (Result): a Result object to combine.
        Returns:
            Result: A new Result object consisting of combined objects.
        """
        ret = copy.deepcopy(self)
        ret += other
        return ret

    def _is_error(self):
        return self._result['status'] == 'ERROR'

    def get_status(self):
        """Return whole qobj result status."""
        return self._result['status']

    def circuit_statuses(self):
        """Return statuses of all circuits

        Returns:
            list(str): List of status result strings.
        """
        return [circuit_result['status']
                for circuit_result in self._result['result']]

    def get_circuit_status(self, icircuit):
        """Return the status of circuit at index icircuit.

        Args:
            icircuit (int): index of circuit
        Returns:
            string: the status of the circuit.
        """
        return self._result['result'][icircuit]['status']

    def get_job_id(self):
        """Return the job id assigned by the api if this is a remote job.

        Returns:
            string: a string containing the job id.
        """
        return self._result['job_id']

    def get_ran_qasm(self, name):
        """Get the ran qasm for the named circuit and backend.

        Args:
            name (str): the name of the quantum circuit.

        Returns:
            string: A text version of the qasm file that has been run.
        Raises:
            QISKitError: if the circuit was not found.
        """
        try:
            qobj = self._qobj
            for index in range(len(qobj["circuits"])):
                if qobj["circuits"][index]['name'] == name:
                    return qobj["circuits"][index]["compiled_circuit_qasm"]
        except KeyError:
            pass
        raise QISKitError('No  qasm for circuit "{0}"'.format(name))

    def get_data(self, circuit_name=None):
        """Get the data of circuit name.

        The data format will depend on the backend. For a real device it
        will be for the form::

            "counts": {'00000': XXXX, '00001': XXXX},
            "time"  : xx.xxxxxxxx

        for the qasm simulators of 1 shot::

            'quantum_state': array([ XXX,  ..., XXX]),
            'classical_state': 0

        for the qasm simulators of n shots::

            'counts': {'0000': XXXX, '1001': XXXX}

        for the unitary simulators::

            'unitary': np.array([[ XX + XXj
                                   ...
                                   XX + XX]
                                 ...
                                 [ XX + XXj
                                   ...
                                   XX + XXj]]

        Args:
            circuit_name (str): the name of the quantum circuit.

        Returns:
            dict: A dictionary of data for the different backends.

        Raises:
            QISKitError: if there is no data for the circuit, or an unhandled
                error occurred while fetching the data.
            Exception: if a handled error occurred while fetching the data.
        """
        if self._is_error():
            exception = self._result['result']
            if isinstance(exception, BaseException):
                raise exception
            else:
                raise QISKitError(str(exception))
        if isinstance(circuit_name, QuantumCircuit):
            circuit_name = circuit_name.name

        if circuit_name is None:
            circuits = list([i['name'] for i in self._qobj['circuits']])
            if len(circuits) == 1:
                circuit_name = circuits[0]
            else:
                raise QISKitError("You have to select a circuit when there is more than"
                                  "one available")

        try:
            qobj = self._qobj
            for index in range(len(qobj['circuits'])):
                if qobj['circuits'][index]['name'] == circuit_name:
                    return self._result['result'][index]['data']
        except (KeyError, TypeError):
            pass
        raise QISKitError('No data for circuit "{0}"'.format(circuit_name))

    def get_counts(self, circuit_name=None):
        """Get the histogram data of circuit name.

        The data from the a qasm circuit is dictionary of the format
        {'00000': XXXX, '00001': XXXXX}.

        Args:
            circuit_name (hashable or None): the name of the quantum circuit.
                If None and there is only one circuit available, returns
                that one.

        Returns:
            Dictionary: Counts {'00000': XXXX, '00001': XXXXX}.

        Raises:
            QISKitError: if there are no counts for the circuit.
        """
        try:
            return self.get_data(circuit_name)['counts']
        except KeyError:
            raise QISKitError('No counts for circuit "{0}"'.format(circuit_name))

    def get_names(self):
        """Get the circuit names of the results.

        Returns:
            List: A list of circuit names.
        """
        return [c['name'] for c in self._qobj['circuits']]

    def average_data(self, name, observable):
        """Compute the mean value of an diagonal observable.

        Takes in an observable in dictionary format and then
        calculates the sum_i value(i) P(i) where value(i) is the value of
        the observable for state i.

        Args:
            name (str): the name of the quantum circuit
            observable (dict): The observable to be averaged over. As an example
            ZZ on qubits equals {"00": 1, "11": 1, "01": -1, "10": -1}

        Returns:
            Double: Average of the observable
        """
        counts = self.get_counts(name)
        temp = 0
        tot = sum(counts.values())
        for key in counts:
            if key in observable:
                temp += counts[key] * observable[key] / tot
        return temp

    def get_qubitpol_vs_xval(self, xvals_dict=None):
        """Compute the polarization of each qubit for all circuits and pull out each circuits
        xval into an array. Assumes that each circuit has the same number of qubits and that
        all qubits are measured.

        Args:
            xvals_dict (dict): xvals for each circuit {'circuitname1': xval1,...}. If this
            is none then the xvals list is just left as an array of zeros

        Returns:
            qubit_pol: mxn double array where m is the number of circuit, n the number of qubits
            xvals: mx1 array of the circuit xvals
        """
        ncircuits = len(self._qobj['circuits'])
        # Is this the best way to get the number of qubits?
        nqubits = self._qobj['circuits'][0]['compiled_circuit']['header']['number_of_qubits']
        qubitpol = numpy.zeros([ncircuits, nqubits], dtype=float)
        xvals = numpy.zeros([ncircuits], dtype=float)

        # build Z operators for each qubit
        z_dicts = []
        for qubit_ind in range(nqubits):
            z_dicts.append(dict())
            for qubit_state in range(2**nqubits):
                new_key = ("{0:0"+"{:d}".format(nqubits) + "b}").format(qubit_state)
                z_dicts[-1][new_key] = -1
                if new_key[nqubits-qubit_ind-1] == '1':
                    z_dicts[-1][new_key] = 1

        # go through each circuit and for eqch qubit and apply the operators using "average_data"
        for circuit_ind in range(ncircuits):
            if xvals_dict:
                xvals[circuit_ind] = xvals_dict[self._qobj['circuits'][circuit_ind]['name']]
            for qubit_ind in range(nqubits):
                qubitpol[circuit_ind, qubit_ind] = self.average_data(
                    self._qobj['circuits'][circuit_ind]['name'], z_dicts[qubit_ind])

        return qubitpol, xvals
