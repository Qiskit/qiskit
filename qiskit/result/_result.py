# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=no-else-return

"""Module for working with Results."""

import logging
import copy
from collections import OrderedDict

import numpy

from qiskit import QISKitError, QuantumCircuit


logger = logging.getLogger(__name__)


class ExperimentResult(object):
    """Container for the results of a single experiment."""

    def __init__(self, qobj_result_experiment):
        """

        Args:
            qobj_result_experiment (qobj.ExperimentResult): schema-conformant
                experiment result.
        """
        self.compiled_circuit_qasm = ''
        self.status = _status_or_success(qobj_result_experiment)
        self.data = qobj_result_experiment.data

    @property
    def counts(self):
        """Histogram of the memory over the indicated shots."""
        return self.data['counts']

    @property
    def snapshots(self):
        """Collection of all snapshots of simulator state."""
        return self.data['snapshots']

    @property
    def statevector(self):
        """Vector of amplitudes describing the simulator state."""
        return self.data['statevector']

    @property
    def unitary(self):
        """Unitary matrix describing the complete evolution of the circuit."""
        return self.data['unitary']


class Result(object):
    """Results for a collection of experiments sent in a ``Qobj`` instance."""

    def __init__(self, qobj_result, experiment_names=None):
        """
        Args:
            qobj_result (qobj.Result): schema-conformant Result.
            experiment_names (list): temporary list of circuit names
        """
        self.backend_name = qobj_result.backend_name
        self.job_id = qobj_result.job_id
        self.status = _status_or_success(qobj_result)

        if experiment_names:
            self.results = OrderedDict(
                zip(experiment_names,
                    [ExperimentResult(i) for i in qobj_result.results]))
        else:
            self.results = OrderedDict(
                (qobj_exp_result.header.get('name'), ExperimentResult(qobj_exp_result))
                for qobj_exp_result in qobj_result.results)

    def __str__(self):
        """Get the status of the run.

        Returns:
            string: the status of the results.
        """
        return self.status

    def __getitem__(self, i):
        return list(self.results.values())[i]

    def __len__(self):
        return len(self.results)

    def __iadd__(self, other):
        """Append a Result object to current Result object.

        Arg:
            other (Result): a Result object to append.
        Returns:
            Result: The current object with appended results.
        Raises:
            QISKitError: if the Results cannot be combined.
        """
        this_backend = self.backend_name
        other_backend = other.backend_name
        if this_backend != other_backend:
            raise QISKitError('Result objects from different backends cannot be combined.')

        if self._is_error() or other._is_error():
            raise QISKitError('Can not combine a failed result with another result.')

        self.results.update(other.results)
        return self

    def __add__(self, other):
        """Combine Result objects.

        Arg:
            other (Result): a Result object to combine.
        Returns:
            Result: A new Result object consisting of combined objects.
        """
        copy_of_self = copy.deepcopy(self)
        copy_of_self += other
        return copy_of_self

    def _is_error(self):
        return self.status in ('ERROR', 'SUCCESS = False')

    def get_status(self):
        """Return whole result status."""
        return self.status

    def circuit_statuses(self):
        """Return statuses of all circuits.

        Returns:
            list(str): List of status result strings.
        """
        return [experiment_result.status for
                experiment_result in self.results.values()]

    def get_circuit_status(self, icircuit):
        """Return the status of circuit at index icircuit.

        Args:
            icircuit (int): index of circuit
        Returns:
            string: the status of the circuit.
        """
        return self[icircuit].status

    def get_job_id(self):
        """Return the job id assigned by the api if this is a remote job.

        Returns:
            string: a string containing the job id.
        """
        return self.job_id

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
            return self.results[name].compiled_circuit_qasm
        except KeyError:
            raise QISKitError('No  qasm for circuit "{0}"'.format(name))

    def get_data(self, circuit=None):
        """Get the data of circuit name.

        The data format will depend on the backend. For a real device it
        will be for the form::

            "counts": {'00000': XXXX, '00001': XXXX},
            "time"  : xx.xxxxxxxx

        for the qasm simulators of 1 shot::

            'statevector': array([ XXX,  ..., XXX]),
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
            circuit (str or QuantumCircuit or None): reference to a quantum circuit
                If None and there is only one circuit available, returns
                that one.

        Returns:
            dict: A dictionary of data for the different backends.

        Raises:
            QISKitError: if there is no data for the circuit, or an unhandled
                error occurred while fetching the data.
        """
        try:
            return self._get_experiment(circuit).data
        except (KeyError, TypeError):
            raise QISKitError('No data for circuit "{0}"'.format(circuit))

    def _get_experiment(self, key=None):
        """Return an experiment from a given key.

        Args:
            key (str or QuantumCircuit or None): reference to a quantum circuit
                If None and there is only one circuit available, returns
                that one.

        Returns:
            ExperimentResult: an Experiment.

        Raises:
            QISKitError: if there is no data for the circuit, or an unhandled
                error occurred while fetching the data.
        """
        if self._is_error():
            raise QISKitError(str(self.status))

        if isinstance(key, QuantumCircuit):
            key = key.name
        elif key is None:
            if len(self.results) != 1:
                raise QISKitError("You have to select a circuit when there is more than "
                                  "one available")
            else:
                key = list(self.results.keys())[0]
        key = str(key)

        return self.results[key]

    def get_counts(self, circuit=None):
        """Get the histogram data of circuit name.

        The data from the a qasm circuit is dictionary of the format
        {'00000': XXXX, '00001': XXXXX}.

        Args:
            circuit (str or QuantumCircuit or None): reference to a quantum circuit
                If None and there is only one circuit available, returns
                that one.

        Returns:
            Dictionary: Counts {'00000': XXXX, '00001': XXXXX}.

        Raises:
            QISKitError: if there are no counts for the circuit.
        """
        try:
            return self._get_experiment(circuit).counts
        except KeyError:
            raise QISKitError('No counts for circuit "{0}"'.format(circuit))

    def get_statevector(self, circuit=None):
        """Get the final statevector of circuit name.

        The data is a list of complex numbers
        [1.+0.j, 0.+0.j].

        Args:
            circuit (str or QuantumCircuit or None): reference to a quantum circuit
                If None and there is only one circuit available, returns
                that one.

        Returns:
            list[complex]: list of 2^n_qubits complex amplitudes.

        Raises:
            QISKitError: if there is no statevector for the circuit.
        """
        try:
            return self._get_experiment(circuit).statevector
        except KeyError:
            raise QISKitError('No statevector for circuit "{0}"'.format(circuit))

    def get_unitary(self, circuit=None):
        """Get the final unitary of circuit name.

        The data is a matrix of complex numbers
        [[1.+0.j, 0.+0.j], .. ].

        Args:
            circuit (str or QuantumCircuit or None): reference to a quantum circuit
                If None and there is only one circuit available, returns
                that one.

        Returns:
            list[list[complex]]: list of 2^n_qubits x 2^n_qubits complex amplitudes.

        Raises:
            QISKitError: if there is no unitary for the circuit.
        """
        try:
            return self._get_experiment(circuit).unitary
        except KeyError:
            raise QISKitError('No unitary for circuit "{0}"'.format(circuit))

    def get_snapshots(self, circuit=None):
        """Get snapshots recorded during the run.

        The data is a dictionary:
        where keys are requested snapshot slots.
        and values are a dictionary of the snapshots themselves.

        Args:
            circuit (str or QuantumCircuit or None): reference to a quantum circuit
                If None and there is only one circuit available, returns
                that one.

        Returns:
            dict[slot: dict[str: array]]: list of 2^n_qubits complex amplitudes.

        Raises:
            QISKitError: if there are no snapshots for the circuit.
        """
        try:
            return self._get_experiment(circuit).snapshots
        except KeyError:
            raise QISKitError('No snapshots for circuit "{0}"'.format(circuit))

    def get_snapshot(self, slot=None, circuit=None):
        """Get snapshot at a specific slot.

        Args:
            slot (str): snapshot slot to retrieve. If None and there is only one
                slot, return that one.
            circuit (str or QuantumCircuit or None): reference to a quantum circuit
                If None and there is only one circuit available, returns
                that one.

        Returns:
            dict[slot: dict[str: array]]: list of 2^n_qubits complex amplitudes.

        Raises:
            QISKitError: if there is no snapshot at all, or in this slot
        """
        try:
            snapshots_dict = self.get_snapshots(circuit)

            if slot is None:
                slots = list(snapshots_dict.keys())
                if len(slots) == 1:
                    slot = slots[0]
                else:
                    raise QISKitError("You have to select a slot when there "
                                      "is more than one available")
            snapshot_dict = snapshots_dict[slot]

            snapshot_types = list(snapshot_dict.keys())
            if len(snapshot_types) == 1:
                snapshot_list = snapshot_dict[snapshot_types[0]]
                if len(snapshot_list) == 1:
                    return snapshot_list[0]
                else:
                    return snapshot_list
            else:
                return snapshot_dict
        except KeyError:
            raise QISKitError('No snapshot at slot {0} for '
                              'circuit "{1}"'.format(slot, circuit))

    def get_names(self):
        """Get the circuit names of the results.

        Returns:
            List: A list of circuit names.
        """
        return list(self.results.keys())

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

    def get_qubitpol_vs_xval(self, nqubits, xvals_dict=None):
        """Compute the polarization of each qubit for all circuits.

        Compute the polarization of each qubit for all circuits and pull out each circuits
        xval into an array. Assumes that each circuit has the same number of qubits and that
        all qubits are measured.

        Args:
            nqubits (int): number of qubits
            xvals_dict (dict): xvals for each circuit {'circuitname1': xval1,...}. If this
            is none then the xvals list is just left as an array of zeros

        Returns:
            qubit_pol: mxn double array where m is the number of circuit, n the number of qubits
            xvals: mx1 array of the circuit xvals
        """
        ncircuits = len(self.results)
        # Is this the best way to get the number of qubits?
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

        # go through each circuit and for each qubit and apply the operators using "average_data"
        for i, (circuit_name, _) in enumerate(self.results.items()):
            if xvals_dict:
                xvals[i] = xvals_dict[circuit_name]
            for qubit_ind in range(nqubits):
                qubitpol[i, qubit_ind] = self.average_data(
                    circuit_name, z_dicts[qubit_ind])

        return qubitpol, xvals


def _status_or_success(obj):
    """Return obj.status or build it from obj.success."""
    # TODO: this is needed because "status" is not a required argument
    # in the schema.
    return getattr(obj, 'status',
                   'SUCCESS = {}'.format(obj.success))
