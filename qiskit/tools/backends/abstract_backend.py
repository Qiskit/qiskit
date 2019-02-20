# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""A module for generating abstract backends."""

from qiskit.test.mock import FakeBackend
from qiskit.providers.models import BackendConfiguration
from qiskit.providers.models.backendconfiguration import GateConfig
from qiskit.qiskiterror import QiskitError


class AbstractBackend(FakeBackend):
    """An abstract backend instance for representing a
    backend with an user defined coupling map."""

    def __init__(self, n_qubits, coupling_map,
                 basis_gates=None,
                 name='AbstractBackend'):
        """
        Args:
            n_qubits (int): Number of qubits.
            coupling_map (list): Coupling map.
            basis_gates (list): Basis gates.
            name (str): Name of backend instance.
        """
        if basis_gates is None:
            basis_gates = ['u1', 'u2', 'u3', 'cx', 'id']

        configuration = BackendConfiguration(
            backend_name=name,
            backend_version='0.0.0',
            n_qubits=n_qubits,
            basis_gates=basis_gates,
            simulator=False,
            local=True,
            conditional=False,
            open_pulse=False,
            memory=False,
            max_shots=8192,
            gates=[GateConfig(name='None', parameters=[], qasm_def='None')],
            coupling_map=coupling_map
        )
        super().__init__(configuration)

    def run(self, _):
        """Run a Qobj on the backend."""
        raise QiskitError('Cannot execute on an AbstractBackend.')
