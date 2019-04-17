# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Circuit transpile function"""

import logging

from qiskit import transpiler
from qiskit.mapper import Layout


logger = logging.getLogger(__name__)


def transpile(circuits, backend=None, basis_gates=None, coupling_map=None,
              backend_properties=None, initial_layout=None,
              seed_transpiler=None, seed_mapper=None):
    """transpile one or more circuits, according to some desired
    transpilation targets.

    Args:
        circuits (QuantumCircuit or list[QuantumCircuit]):
            Circuit(s) to transpile

        backend (BaseBackend):
            If set, transpiler options are automatically grabbed from
            backend.configuration() and backend.properties().
            If any other option is explicitly set (e.g. coupling_map), it
            will override the backend's.
            Note: the backend arg is purely for convenience. The resulting
                circuit may be run on any backend as long as it is compatible.

        basis_gates (list[str]):
            List of basis gate names to unroll to.
            e.g:
                ['u1', 'u2', 'u3', 'cx']
            If None, do not unroll.

        coupling_map (list):
            Coupling map (perhaps custom) to target in mapping.
            Must be given as an adjacency matrix, where each entry
            specifies all two-qubit interactions supported by backend
            e.g:
                [[0, 1], [0, 3], [1, 2], [1, 5], [2, 5], [4, 1], [5, 3]]

        backend_properties (BackendProperties):
            properties returned by a backend, including information on gate
            errors, readout errors, qubit coherence times, etc. For a backend
            that provides this information, it can be obtained with:
            ``backend.properties()``

        initial_layout (list or dict):
            Initial layout of virtual qubits onto the physical qubits
            If this layout makes the circuit compatible with the coupling_map
            constraints, it will be used.
            The final layout will not necessarily be the same, since swaps may
            permute the location of qubits.
            e.g:
                [q[0], None, None, q[1], None]
                {q[0]: 0, q[1]: 3}
            N.B. These layouts are not identical. In the first case, it is
            signaled to the transpiler that three unspecified virtual qubits
            must exist on physical qubits 1, 2 and 4. The transpiler will
            expand the original circuit with extra virtual ancilla qubits
            to accomodate this layout.

        seed_transpiler (int):
            sets random seed for the stochastic parts of the transpiler

        seed_mapper (int):
            DEPRECATED in 0.8: use ``seed_transpiler`` kwarg instead

        pass_manager (PassManager):
            DEPRECATED in 0.8: use ``pass_manager.run(circuits)`` directly

    Returns:
        QuantumCircuit or list[QuantumCircuit]: transpiled circuit(s).
    """

    # ------------
    # TODO: This is a hack while we are still using the old transpiler.
    initial_layout = getattr(transpile_config, 'initial_layout', None)
    basis_gates = getattr(transpile_config, 'basis_gates', None)
    coupling_map = getattr(transpile_config, 'coupling_map', None)
    seed_mapper = getattr(transpile_config, 'seed_mapper', None)

    if initial_layout is not None and not isinstance(initial_layout, Layout):
        initial_layout = Layout(initial_layout)

    pass_manager = None
    backend = getattr(transpile_config, 'backend', None)
    new_circuits = transpiler.transpile(circuits, backend, basis_gates, coupling_map,
                                        initial_layout, seed_mapper, pass_manager)
    # ---------

    # THE IDEAL CODE HERE WILL BE.
    # 1 set up the pass_manager using transpile_config options
    # pass_manager = PassManager(TranspileConig)
    # run the passes
    # new_circuits = pass_manager.run(circuits)
    return new_circuits
