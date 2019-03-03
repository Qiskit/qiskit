# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Circuit transpile function """
import logging

from qiskit import transpiler
from qiskit.mapper import Layout


logger = logging.getLogger(__name__)


def transpile(circuits, transpile_config=None):
    """Compile a list of circuits into a list of optimized circuits.

    Args:
        circuits (QuantumCircuit or list[QuantumCircuit]): circuits to compile
        transpile_config (TranspileConfig): configuration for the transpiler

    Returns:
        circuits: the optimized circuits

    """

    # ------------
    # This is a HACK while we are still using the old transpiler.
    try:
        initial_layout = transpile_config.initial_layout
    except AttributeError:
        initial_layout = None
    try:
        basis_gates = transpile_config.basis_gates
    except AttributeError:
        basis_gates = None
    try:
        coupling_map = transpile_config.coupling_map
    except AttributeError:
        coupling_map = None
    try:
        seed_mapper = transpile_config.seed_mapper
    except AttributeError:
        seed_mapper = None

    if initial_layout is not None and not isinstance(initial_layout, Layout):
        initial_layout = Layout(initial_layout)

    pass_manager = None
    backend = transpile_config.backend
    new_circuits = transpiler.transpile(circuits, backend, basis_gates, coupling_map,
                                        initial_layout, seed_mapper, pass_manager)
    # ---------

    # THE IDEAL CODE HERE WILL BE.
    # 1 set up the pass_manager from transconfig
    # pass_manager = PassManager(TranspileConig)
    # run the passes
    # new_circuits = pass_manager.run(circuits)
    return new_circuits
