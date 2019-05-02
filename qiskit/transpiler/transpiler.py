# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=too-many-function-args, unexpected-keyword-arg

"""THIS FILE IS DEPRECATED AND WILL BE REMOVED IN RELEASE 0.9.
"""
import warnings

from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.transpiler import CouplingMap
from qiskit import compiler
from qiskit.transpiler.preset_passmanagers import (default_pass_manager_simulator,
                                                   default_pass_manager)


def transpile(circuits, backend=None, basis_gates=None, coupling_map=None,
              initial_layout=None, seed_mapper=None, pass_manager=None):
    """transpile one or more circuits.

    Args:
        circuits (QuantumCircuit or list[QuantumCircuit]): circuits to compile
        backend (BaseBackend): a backend to compile for
        basis_gates (list[str]): list of basis gate names supported by the
            target. Default: ['u1','u2','u3','cx','id']
        coupling_map (list): coupling map (perhaps custom) to target in mapping

        initial_layout (Layout or dict or list):
            Initial position of virtual qubits on physical qubits. The final
            layout is not guaranteed to be the same, as the transpiler may permute
            qubits through swaps or other means.

        seed_mapper (int): random seed for the swap_mapper
        pass_manager (PassManager): a pass_manager for the transpiler stages

    Returns:
        QuantumCircuit or list[QuantumCircuit]: transpiled circuit(s).

    Raises:
        TranspilerError: in case of bad inputs to transpiler or errors in passes
    """
    warnings.warn("qiskit.transpiler.transpile() has been deprecated and will be "
                  "removed in the 0.9 release. Use qiskit.compiler.transpile() instead.",
                  DeprecationWarning)
    return compiler.transpile(circuits=circuits, backend=backend,
                              basis_gates=basis_gates, coupling_map=coupling_map,
                              initial_layout=initial_layout, seed_transpiler=seed_mapper,
                              pass_manager=pass_manager)


def transpile_dag(dag, basis_gates=None, coupling_map=None,
                  initial_layout=None, seed_mapper=None, pass_manager=None):
    """Deprecated - Use qiskit.compiler.transpile for transpiling from
    circuits to circuits.
    Transform a dag circuit into another dag circuit
    (transpile), through consecutive passes on the dag.

    Args:
        dag (DAGCircuit): dag circuit to transform via transpilation
        basis_gates (list[str]): list of basis gate names supported by the
            target. Default: ['u1','u2','u3','cx','id']
        coupling_map (list): A graph of coupling::

            [
             [control0(int), target0(int)],
             [control1(int), target1(int)],
            ]

            eg. [[0, 2], [1, 2], [1, 3], [3, 4]}

        initial_layout (Layout or None): A layout object
        seed_mapper (int): random seed_mapper for the swap mapper
        pass_manager (PassManager): pass manager instance for the transpilation process
            If None, a default set of passes are run.
            Otherwise, the passes defined in it will run.
            If contains no passes in it, no dag transformations occur.

    Returns:
        DAGCircuit: transformed dag
    """
    warnings.warn("transpile_dag has been deprecated and will be removed in the "
                  "0.9 release. Circuits can be transpiled directly to other "
                  "circuits with the transpile function.", DeprecationWarning)

    if basis_gates is None:
        basis_gates = ['u1', 'u2', 'u3', 'cx', 'id']

    if pass_manager is None:
        # default set of passes

        # if a coupling map is given compile to the map
        if coupling_map:
            pass_manager = default_pass_manager(basis_gates,
                                                CouplingMap(coupling_map),
                                                initial_layout,
                                                seed_transpiler=seed_mapper)
        else:
            pass_manager = default_pass_manager_simulator(basis_gates)

    # run the passes specified by the pass manager
    # TODO return the property set too. See #1086
    name = dag.name
    circuit = dag_to_circuit(dag)
    circuit = pass_manager.run(circuit)
    dag = circuit_to_dag(circuit)
    dag.name = name

    return dag
