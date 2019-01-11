# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Model for schema-conformant Results."""

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError

from qiskit.validation.base import BaseModel, bind_schema
from .postprocess import (format_counts, format_statevector,
                          format_unitary, format_memory)
from .models import ResultSchema


@bind_schema(ResultSchema)
class Result(BaseModel):
    """Model for Results.

    Please note that this class only describes the required fields. For the
    full description of the model, please check ``ResultSchema``.

    Attributes:
        backend_name (str): backend name.
        backend_version (str): backend version, in the form X.Y.Z.
        qobj_id (str): user-generated Qobj id.
        job_id (str): unique execution id from the backend.
        success (bool): True if complete input qobj executed correctly. (Implies
            each experiment success)
        results (ExperimentResult): corresponding results for array of
            experiments of the input qobj
    """

    def __init__(self, backend_name, backend_version, qobj_id, job_id, success,
                 results, **kwargs):
        self.backend_name = backend_name
        self.backend_version = backend_version
        self.qobj_id = qobj_id
        self.job_id = job_id
        self.success = success
        self.results = results

        super().__init__(**kwargs)

    def data(self, circuit=None):
        """Get the raw data for an experiment.

        Note this data will be a single classical and quantum register and in a
        format required by the results schema. We recomened that most users use
        the get_xxx method, and the data will be post-processed for the data type.

        Args:
            circuit (str or QuantumCircuit or int or None): the index of the
                experiment. Several types are accepted for convenience::
                * str: the name of the experiment.
                * QuantumCircuit: the name of the instance will be used.
                * int: the position of the experiment.
                * None: if there is only one experiment, returns it.

        Returns:
            dict: A dictionary of results data for an experiment. The data
            depends on the backend it ran on.

            QASM backends return a dictionary of dictionary with the key
            'counts' and  with the counts, with the second dictionary keys
            containing a string in hex format (``0x123``) and values equal to
            the number of times this outcome was measured.

            Statevector backends return a dictionary with key 'statevector' and
            values being a list[list[complex components]] list of 2^n_qubits
            complex amplitudes. Where each complex number is represented as a 2
            entry list for each component. For example, a list of
            [0.5+1j, 0-1j] would be represented as [[0.5, 1], [0, -1]].

            Unitary backends return a dictionary with key 'unitary' and values
            being a list[list[list[complex components]]] list of
            2^n_qubits x 2^n_qubits complex amplitudes in a two entry list for
            each component. For example if the amplitude is
            [[0.5+0j, 0-1j], ...] the value returned will be
            [[[0.5, 0], [0, -1]], ...].

            The simulator backends also have an optional key 'snapshots' which
            returns a dict of snapshots specified by the simulator backend.
            The value is of the form dict[slot: dict[str: array]]
            where the keys are the requested snapshot slots, and the values are
            a dictionary of the snapshots.

        Raises:
            QiskitError: if data for the experiment could not be retrieved.
        """
        try:
            return self._get_experiment(circuit).data.to_dict()
        except (KeyError, TypeError):
            raise QiskitError('No data for circuit "{0}"'.format(circuit))

    def get_memory(self, circuit=None):
        """Get the sequence of memory states (readouts) for each shot
        The data from the experiment is a list of format
        ['00000', '01000', '10100', '10100', '11101', '11100', '00101', ..., '01010']

        Args:
            circuit (str or QuantumCircuit or int or None): the index of the
                experiment, as specified by ``data()``.

        Returns:
            List[str]: the list of each outcome, formatted according to
                registers in circuit.

        Raises:
            QiskitError: if there is no memory data for the circuit.
        """
        try:
            header = self._get_experiment(circuit).header.to_dict()
            memory_list = []
            for memory in self.data(circuit)['memory']:
                memory_list.append(format_memory(memory, header))
            return memory_list
        except KeyError:
            raise QiskitError('No memory for circuit "{0}".'.format(circuit))

    def get_counts(self, circuit=None):
        """Get the histogram data of an experiment.

        Args:
            circuit (str or QuantumCircuit or int or None): the index of the
                experiment, as specified by ``get_data()``.

        Returns:
            dict[str:int]: a dictionary with the counts for each qubit, with
                the keys containing a string in binary format and separated
                according to the registers in circuit (e.g. ``0100 1110``).
                The string is little-endian (cr[0] on the right hand side).

        Raises:
            QiskitError: if there are no counts for the experiment.
        """
        try:
            return format_counts(self.data(circuit)['counts'],
                                 self._get_experiment(circuit).header.to_dict())
        except KeyError:
            raise QiskitError('No counts for circuit "{0}"'.format(circuit))

    def get_statevector(self, circuit=None, decimals=None):
        """Get the final statevector of an experiment.

        Args:
            circuit (str or QuantumCircuit or int or None): the index of the
                experiment, as specified by ``data()``.
            decimals (int): the number of decimals in the statevector.
                If None, does not round.

        Returns:
            list[complex]: list of 2^n_qubits complex amplitudes.

        Raises:
            QiskitError: if there is no statevector for the experiment.
        """
        try:
            return format_statevector(self.data(circuit)['statevector'],
                                      decimals=decimals)
        except KeyError:
            raise QiskitError('No statevector for circuit "{0}"'.format(circuit))

    def get_unitary(self, circuit=None, decimals=None):
        """Get the final unitary of an experiment.

        Args:
            circuit (str or QuantumCircuit or int or None): the index of the
                experiment, as specified by ``data()``.
            decimals (int): the number of decimals in the unitary.
                If None, does not round.

        Returns:
            list[list[complex]]: list of 2^n_qubits x 2^n_qubits complex
                amplitudes.

        Raises:
            QiskitError: if there is no unitary for the experiment.
        """
        try:
            return format_unitary(self.data(circuit)['unitary'],
                                  decimals=decimals)
        except KeyError:
            raise QiskitError('No unitary for circuit "{0}"'.format(circuit))

    def _get_experiment(self, key=None):
        """Return a single experiment result from a given key.

        Args:
            key (str or QuantumCircuit or int or None): the index of the
                experiment, as specified by ``get_data()``.

        Returns:
            ExperimentResult: the results for an experiment.

        Raises:
            QiskitError: if there is no data for the circuit, or an unhandled
                error occurred while fetching the data.
        """
        if not self.success:
            raise QiskitError(getattr(self, 'status',
                                      'Result was not successful'))

        # Automatically return the first result if no key was provided.
        if key is None:
            if len(self.results) != 1:
                raise QiskitError(
                    'You have to select a circuit when there is more than '
                    'one available')
            else:
                key = 0

        # Key is an integer: return result by index.
        if isinstance(key, int):
            return self.results[key]

        # Key is a QuantumCircuit or str: retrieve result by name.
        if isinstance(key, QuantumCircuit):
            key = key.name

        try:
            # Look into `result[x].header.name` for the names.
            return next(result for result in self.results
                        if getattr(getattr(result, 'header', None),
                                   'name', '') == key)
        except StopIteration:
            raise QiskitError('Data for experiment "%s" could not be found.' %
                              key)
