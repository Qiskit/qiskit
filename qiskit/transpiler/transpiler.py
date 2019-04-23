# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tools for compiling a batch of quantum circuits."""
import logging
import warnings

from qiskit.circuit import QuantumCircuit
from qiskit.mapper import CouplingMap
from qiskit.tools.parallel import parallel_map
from qiskit.converters import circuit_to_dag
from qiskit.converters import dag_to_circuit
from qiskit.mapper.layout import Layout
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.preset_passmanagers import (default_pass_manager_simulator,
                                                   default_pass_manager)

logger = logging.getLogger(__name__)


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

            Multiple formats are supported:
            a. Layout instance

            b. dict
                virtual to physical:
                    {qr[0]: 0,
                     qr[1]: 3,
                     qr[2]: 5}

                physical to virtual:
                    {0: qr[0],
                     3: qr[1],
                     5: qr[2]}

            c. list
                virtual to physical:
                    [0, 3, 5]  # virtual qubits are ordered (in addition to named)

                physical to virtual:
                    [qr[0], None, None, qr[1], None, qr[2]]


        seed_mapper (int): random seed for the swap_mapper
        pass_manager (PassManager): a pass_manager for the transpiler stages

    Returns:
        QuantumCircuit or list[QuantumCircuit]: transpiled circuit(s).

    Raises:
        TranspilerError: in case of bad inputs to transpiler or errors in passes
    """
    return_form_is_single = False
    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]
        return_form_is_single = True

    # pass manager overrides explicit transpile options (basis_gates, coupling_map)
    # explicit transpile options override options gotten from a backend
    if not pass_manager and backend:
        basis_gates = basis_gates or getattr(backend.configuration(), 'basis_gates',
                                             ['u1', 'u2', 'u3', 'id', 'cx'])
        # This needs to be removed once Aer 0.2 is out
        coupling_map = coupling_map or getattr(backend.configuration(), 'coupling_map', None)

    # Accomodate initial_layout in the form of Layout, or dict, or list
    # TODO: (Ali) allow partial layout specifications
    if isinstance(initial_layout, list):
        if all(isinstance(elem, int) for elem in initial_layout):
            # FIXME: (Ali) must allow a list of initial layouts
            initial_layout = Layout.from_intlist(initial_layout, *circuits[0].qregs)
        elif all(elem is None or isinstance(elem, tuple) for elem in initial_layout):
            initial_layout = Layout.from_tuplelist(initial_layout)
        else:
            raise TranspilerError("bad format for initial_layout")
    elif isinstance(initial_layout, dict):
        initial_layout = Layout(initial_layout)

    circuits = parallel_map(_transpilation, circuits,
                            task_kwargs={'basis_gates': basis_gates,
                                         'coupling_map': coupling_map,
                                         'initial_layout': initial_layout,
                                         'seed_mapper': seed_mapper,
                                         'pass_manager': pass_manager})
    if return_form_is_single:
        return circuits[0]
    return circuits


def _transpilation(circuit, basis_gates=None, coupling_map=None,
                   initial_layout=None, seed_mapper=None,
                   pass_manager=None):
    """Perform transpilation of a single circuit.

    Args:
        circuit (QuantumCircuit): A circuit to transpile.
        basis_gates (list[str]): list of basis gate names supported by the
            target. Default: ['u1','u2','u3','cx','id']
        coupling_map (CouplingMap): coupling map (perhaps custom) to target in mapping
        initial_layout (Layout): initial layout of qubits in mapping
        seed_mapper (int): random seed for the swap_mapper
        pass_manager (PassManager): a pass_manager for the transpiler stage

    Returns:
        QuantumCircuit: A transpiled circuit.

    Raises:
        TranspilerError: If the Layout does not matches the circuit
    """
    if initial_layout is not None and set(circuit.qregs) != initial_layout.get_registers():
        raise TranspilerError('The provided initial layout does not match the registers in '
                              'the circuit "%s"' % circuit.name)

    if pass_manager and not pass_manager.working_list:
        return circuit

    is_parametric_circuit = bool(circuit.unassigned_variables)

    dag = circuit_to_dag(circuit)
    del circuit

    final_dag = _transpile_dag(dag, basis_gates=basis_gates,
                               coupling_map=coupling_map,
                               initial_layout=initial_layout,
                               skip_numeric_passes=is_parametric_circuit,
                               seed_mapper=seed_mapper,
                               pass_manager=pass_manager)

    out_circuit = dag_to_circuit(final_dag)

    return out_circuit


def transpile_dag(dag, basis_gates=None, coupling_map=None,
                  initial_layout=None, skip_numeric_passes=None,
                  seed_mapper=None, pass_manager=None):
    """Deprecated - Transform a dag circuit into another dag circuit
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
        skip_numeric_passes (bool): If true, skip passes which require fixed parameter values
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
    return _transpile_dag(dag, basis_gates, coupling_map, initial_layout,
                          skip_numeric_passes, seed_mapper, pass_manager)


def _transpile_dag(dag, basis_gates=None, coupling_map=None,
                   initial_layout=None, skip_numeric_passes=None,
                   seed_mapper=None, pass_manager=None):
    """Transform a dag circuit into another dag circuit (transpile), through
    consecutive passes on the dag.

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
        skip_numeric_passes (bool): If true, skip passes which require fixed parameter values
        seed_mapper (int): random seed_mapper for the swap mapper
        pass_manager (PassManager): pass manager instance for the transpilation process
            If None, a default set of passes are run.
            Otherwise, the passes defined in it will run.
            If contains no passes in it, no dag transformations occur.

    Returns:
        DAGCircuit: transformed dag
    """
    # TODO: `basis_gates` will be removed after we have the unroller pass.
    # TODO: `coupling_map`, `initial_layout`, `seed_mapper` removed after mapper pass.

    if basis_gates is None:
        basis_gates = ['u1', 'u2', 'u3', 'cx', 'id']
    if isinstance(basis_gates, str):
        warnings.warn("The parameter basis_gates is now a list of strings. "
                      "For example, this basis ['u1','u2','u3','cx'] should be used "
                      "instead of 'u1,u2,u3,cx'. The string format will be "
                      "removed after 0.9", DeprecationWarning, 2)
        basis_gates = basis_gates.split(',')

    if pass_manager is None:
        # default set of passes

        # if a coupling map is given compile to the map
        if coupling_map:
            pass_manager = default_pass_manager(basis_gates,
                                                CouplingMap(coupling_map),
                                                initial_layout,
                                                skip_numeric_passes,
                                                seed_mapper=seed_mapper)
        else:
            pass_manager = default_pass_manager_simulator(basis_gates)

    # run the passes specified by the pass manager
    # TODO return the property set too. See #1086
    name = dag.name
    dag = pass_manager.run_passes(dag)
    dag.name = name

    return dag
