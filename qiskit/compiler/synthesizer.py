# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Circuit synthesize function """
import warnings
import logging

from qiskit import transpiler
from qiskit.compiler import TranspileConfig
from qiskit.mapper import Layout


logger = logging.getLogger(__name__)


def synthesize_circuits(circuits, transpile_config=None, pass_manager=None):
    """Compile a list of circuits into a qobj.

    Args:
        circuits (QuantumCircuit or list[QuantumCircuit]): circuits to compile
        transpile_config (TranspileConfig): configuration for the transpiler
        pass_manager (PassManager): a pass manger for the transpiler pipeline

    Returns:
        circuits (QuantumCircuit or list[QuantumCircuit]: the synthesized circuits

    """

    # ------------
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

    # This is a HACK while we are still using the old transpiler.
    backend = transpile_config.backend
    new_circuits = transpiler.transpile(circuits, backend, basis_gates, coupling_map,
                                        initial_layout, seed_mapper, pass_manager)
    # ---------

    # THE IDEAL CODE HERE WILL

    # USING THE PASS MANAGER TO
    return new_circuits
