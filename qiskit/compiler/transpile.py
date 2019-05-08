# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Circuit transpile function"""
import warnings

from qiskit.transpiler import Layout, CouplingMap
from qiskit.tools.parallel import parallel_map
from qiskit.transpiler.transpile_config import TranspileConfig
from qiskit.transpiler.transpile_circuit import transpile_circuit
from qiskit.pulse import Schedule


def transpile(circuits,
              backend=None,
              basis_gates=None, coupling_map=None, backend_properties=None,
              initial_layout=None, seed_transpiler=None,
              optimization_level=None,
              pass_manager=None):
    """transpile one or more circuits, according to some desired
    transpilation targets.

    All arguments may be given as either singleton or list. In case of list,
    the length must be equal to the number of circuits being transpiled.

    Transpilation is done in parallel using multiprocessing.

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

        coupling_map (CouplingMap or list):
            Coupling map (perhaps custom) to target in mapping.
            Multiple formats are supported:
            a. CouplingMap instance

            b. list
                Must be given as an adjacency matrix, where each entry
                specifies all two-qubit interactions supported by backend
                e.g:
                    [[0, 1], [0, 3], [1, 2], [1, 5], [2, 5], [4, 1], [5, 3]]

        backend_properties (BackendProperties):
            properties returned by a backend, including information on gate
            errors, readout errors, qubit coherence times, etc. For a backend
            that provides this information, it can be obtained with:
            ``backend.properties()``

        initial_layout (Layout or dict or list):
            Initial position of virtual qubits on physical qubits.
            If this layout makes the circuit compatible with the coupling_map
            constraints, it will be used.
            The final layout is not guaranteed to be the same, as the transpiler
            may permute qubits through swaps or other means.

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

        seed_transpiler (int):
            sets random seed for the stochastic parts of the transpiler

        optimization_level (int):
            How much optimization to perform on the circuits.
            Higher levels generate more optimized circuits,
            at the expense of longer transpilation time.
                0: no optimization
                1: light optimization
                2: heavy optimization

        pass_manager (PassManager):
            The pass manager to use for a custom pipeline of transpiler passes.
            If this arg is present, all other args will be ignored and the
            pass manager will be used directly (Qiskit will not attempt to
            auto-select a pass manager based on transpile options).


    Returns:
        QuantumCircuit or list[QuantumCircuit]: transpiled circuit(s).

    Raises:
        TranspilerError: in case of bad inputs to transpiler or errors in passes
    """

    # transpiling schedules is not supported yet.
    if isinstance(circuits, Schedule) or \
       (isinstance(circuits, list) and all(isinstance(c, Schedule) for c in circuits)):
        return circuits

    # Get TranspileConfig(s) to configure the circuit transpilation job(s)
    circuits = circuits if isinstance(circuits, list) else [circuits]
    transpile_configs = _parse_transpile_args(circuits, backend, basis_gates, coupling_map,
                                              backend_properties, initial_layout,
                                              seed_transpiler, optimization_level,
                                              pass_manager)

    # Transpile circuits in parallel
    circuits = parallel_map(_transpile_circuit, list(zip(circuits, transpile_configs)))

    if len(circuits) == 1:
        return circuits[0]
    return circuits


# FIXME: This is a helper function because of parallel tools.
def _transpile_circuit(circuit_config_tuple):
    """Select a PassManager and run a single circuit through it.

    Args:
        circuit_config_tuple (tuple):
            circuit (QuantumCircuit): circuit to transpile
            transpile_config (TranspileConfig): configuration dictating how to transpile

    Returns:
        QuantumCircuit: transpiled circuit
    """
    circuit, transpile_config = circuit_config_tuple

    return transpile_circuit(circuit, transpile_config)


def _parse_transpile_args(circuits, backend,
                          basis_gates, coupling_map, backend_properties,
                          initial_layout, seed_transpiler, optimization_level,
                          pass_manager):
    """Resolve the various types of args allowed to the transpile() function through
    duck typing, overriding args, etc. Refer to the transpile() docstring for details on
    what types of inputs are allowed.

    Here the args are resolved by converting them to standard instances, and prioritizing
    them in case a transpile option is passed through multiple args (explicitly setting an
    arg has more priority than the arg set by backend)

    Returns:
        list[TranspileConfig]: a transpile config for each circuit, which is a standardized
            object that configures the transpiler and determines the pass manager to use.
    """
    # Each arg could be single or a list. If list, it must be the same size as
    # number of circuits. If single, duplicate to create a list of that size.
    num_circuits = len(circuits)

    basis_gates = _parse_basis_gates(basis_gates, backend, circuits)

    coupling_map = _parse_coupling_map(coupling_map, backend, num_circuits)

    backend_properties = _parse_backend_properties(backend_properties, backend, num_circuits)

    initial_layout = _parse_initial_layout(initial_layout, circuits)

    seed_transpiler = _parse_seed_transpiler(seed_transpiler, num_circuits)

    optimization_level = _parse_optimization_level(optimization_level, num_circuits)

    pass_manager = _parse_pass_manager(pass_manager, num_circuits)

    transpile_configs = []
    for args in zip(basis_gates, coupling_map, backend_properties, initial_layout,
                    seed_transpiler, optimization_level, pass_manager):
        transpile_config = TranspileConfig(basis_gates=args[0],
                                           coupling_map=args[1],
                                           backend_properties=args[2],
                                           initial_layout=args[3],
                                           seed_transpiler=args[4],
                                           optimization_level=args[5],
                                           pass_manager=args[6])
        transpile_configs.append(transpile_config)

    return transpile_configs


def _parse_basis_gates(basis_gates, backend, circuits):
    # try getting basis_gates from user, else backend
    if basis_gates is None:
        if getattr(backend, 'configuration', None):
            basis_gates = getattr(backend.configuration(), 'basis_gates', None)
    # basis_gates could be None, or a list of basis, e.g. ['u3', 'cx']
    if isinstance(basis_gates, str):
        warnings.warn("The parameter basis_gates is now a list of strings. "
                      "For example, this basis ['u1','u2','u3','cx'] should be used "
                      "instead of 'u1,u2,u3,cx'. The string format will be "
                      "removed after 0.9", DeprecationWarning, 2)
        basis_gates = basis_gates.split(',')
    if basis_gates is None or (isinstance(basis_gates, list) and
                               all(isinstance(i, str) for i in basis_gates)):
        basis_gates = [basis_gates] * len(circuits)

    # no basis means don't unroll (all circuit gates are valid basis)
    for index, circuit in enumerate(circuits):
        basis = basis_gates[index]
        if basis is None:
            gates_in_circuit = set(inst.name for inst, _, _ in circuit.data)
            # Other passes might add new gates that need to be supported
            basis_gates[index] = list(gates_in_circuit.union(['u3', 'cx']))
    return basis_gates


def _parse_coupling_map(coupling_map, backend, num_circuits):
    # try getting coupling_map from user, else backend
    if coupling_map is None:
        if getattr(backend, 'configuration', None):
            coupling_map = getattr(backend.configuration(), 'coupling_map', None)
    # coupling_map could be None, or a list of lists, e.g. [[0, 1], [2, 1]]
    if coupling_map is None or isinstance(coupling_map, CouplingMap):
        coupling_map = [coupling_map] * num_circuits
    elif isinstance(coupling_map, list) and all(isinstance(i, list) and len(i) == 2
                                                for i in coupling_map):
        coupling_map = [coupling_map] * num_circuits
    coupling_map = [CouplingMap(cm) if isinstance(cm, list) else cm for cm in coupling_map]
    return coupling_map


def _parse_backend_properties(backend_properties, backend, num_circuits):
    # try getting backend_properties from user, else backend
    if backend_properties is None:
        if getattr(backend, 'properties', None):
            backend_properties = backend.properties()
    if not isinstance(backend_properties, list):
        backend_properties = [backend_properties] * num_circuits
    return backend_properties


def _parse_initial_layout(initial_layout, circuits):
    # initial_layout could be None, or a list of ints, e.g. [0, 5, 14]
    # or a list of tuples/None e.g. [qr[0], None, qr[1]] or a dict e.g. {qr[0]: 0}
    def _layout_from_raw(initial_layout, circuit):
        if isinstance(initial_layout, list):
            if all(isinstance(elem, int) for elem in initial_layout):
                initial_layout = Layout.from_intlist(initial_layout, *circuit.qregs)
            elif all(elem is None or isinstance(elem, tuple) for elem in initial_layout):
                initial_layout = Layout.from_tuplelist(initial_layout)
        elif isinstance(initial_layout, dict):
            initial_layout = Layout(initial_layout)
        return initial_layout
    # multiple layouts?
    if isinstance(initial_layout, list) and \
       any(isinstance(i, (list, dict)) for i in initial_layout):
        initial_layout = [_layout_from_raw(lo, circ) if isinstance(lo, (list, dict)) else lo
                          for lo, circ in zip(initial_layout, circuits)]
    else:
        # even if one layout, but multiple circuits, the layout needs to be adapted for each
        initial_layout = [_layout_from_raw(initial_layout, circ) for circ in circuits]
    if not isinstance(initial_layout, list):
        initial_layout = [initial_layout] * len(circuits)
    return initial_layout


def _parse_seed_transpiler(seed_transpiler, num_circuits):
    if not isinstance(seed_transpiler, list):
        seed_transpiler = [seed_transpiler] * num_circuits
    return seed_transpiler


def _parse_optimization_level(optimization_level, num_circuits):
    if not isinstance(optimization_level, list):
        optimization_level = [optimization_level] * num_circuits
    return optimization_level


def _parse_pass_manager(pass_manager, num_circuits):
    if not isinstance(pass_manager, list):
        pass_manager = [pass_manager] * num_circuits
    return pass_manager
