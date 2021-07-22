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
import logging
import warnings
from time import time
from typing import List, Union, Dict, Callable, Any, Optional, Tuple

from qiskit import user_config
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import Qubit
from qiskit.converters import isinstanceint, isinstancelist, dag_to_circuit, circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.providers import BaseBackend
from qiskit.providers.backend import Backend
from qiskit.providers.models import BackendProperties
from qiskit.providers.models.backendproperties import Gate
from qiskit.pulse import Schedule
from qiskit.tools.parallel import parallel_map
from qiskit.transpiler import Layout, CouplingMap, PropertySet, PassManager
from qiskit.transpiler.basepasses import BasePass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations, InstructionDurationsType
from qiskit.transpiler.passes import ApplyLayout
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.timing_constraints import TimingConstraints
from qiskit.transpiler.preset_passmanagers import (
    level_0_pass_manager,
    level_1_pass_manager,
    level_2_pass_manager,
    level_3_pass_manager,
)

logger = logging.getLogger(__name__)


def transpile(
    circuits: Union[QuantumCircuit, List[QuantumCircuit]],
    backend: Optional[Union[Backend, BaseBackend]] = None,
    basis_gates: Optional[List[str]] = None,
    coupling_map: Optional[Union[CouplingMap, List[List[int]]]] = None,
    backend_properties: Optional[BackendProperties] = None,
    initial_layout: Optional[Union[Layout, Dict, List]] = None,
    layout_method: Optional[str] = None,
    routing_method: Optional[str] = None,
    translation_method: Optional[str] = None,
    scheduling_method: Optional[str] = None,
    instruction_durations: Optional[InstructionDurationsType] = None,
    dt: Optional[float] = None,
    approximation_degree: Optional[float] = None,
    timing_constraints: Optional[Dict[str, int]] = None,
    seed_transpiler: Optional[int] = None,
    optimization_level: Optional[int] = None,
    pass_manager: Optional[PassManager] = None,
    callback: Optional[Callable[[BasePass, DAGCircuit, float, PropertySet, int], Any]] = None,
    output_name: Optional[Union[str, List[str]]] = None,
) -> Union[QuantumCircuit, List[QuantumCircuit]]:
    """Transpile one or more circuits, according to some desired transpilation targets.

    All arguments may be given as either a singleton or list. In case of a list,
    the length must be equal to the number of circuits being transpiled.

    Transpilation is done in parallel using multiprocessing.

    Args:
        circuits: Circuit(s) to transpile
        backend: If set, transpiler options are automatically grabbed from
            ``backend.configuration()`` and ``backend.properties()``.
            If any other option is explicitly set (e.g., ``coupling_map``), it
            will override the backend's.

            .. note::

                The backend arg is purely for convenience. The resulting
                circuit may be run on any backend as long as it is compatible.
        basis_gates: List of basis gate names to unroll to
            (e.g: ``['u1', 'u2', 'u3', 'cx']``). If ``None``, do not unroll.
        coupling_map: Coupling map (perhaps custom) to target in mapping.
            Multiple formats are supported:

            #. ``CouplingMap`` instance
            #. List, must be given as an adjacency matrix, where each entry
               specifies all two-qubit interactions supported by backend,
               e.g: ``[[0, 1], [0, 3], [1, 2], [1, 5], [2, 5], [4, 1], [5, 3]]``

        backend_properties: properties returned by a backend, including information on gate
            errors, readout errors, qubit coherence times, etc. Find a backend
            that provides this information with: ``backend.properties()``
        initial_layout: Initial position of virtual qubits on physical qubits.
            If this layout makes the circuit compatible with the coupling_map
            constraints, it will be used. The final layout is not guaranteed to be the same,
            as the transpiler may permute qubits through swaps or other means.
            Multiple formats are supported:

            #. ``Layout`` instance
            #. Dict
               * virtual to physical::

                    {qr[0]: 0,
                     qr[1]: 3,
                     qr[2]: 5}

               * physical to virtual::

                    {0: qr[0],
                     3: qr[1],
                     5: qr[2]}

            #. List

               * virtual to physical::

                    [0, 3, 5]  # virtual qubits are ordered (in addition to named)

               * physical to virtual::

                    [qr[0], None, None, qr[1], None, qr[2]]

        layout_method: Name of layout selection pass ('trivial', 'dense', 'noise_adaptive', 'sabre')
        routing_method: Name of routing pass ('basic', 'lookahead', 'stochastic', 'sabre', 'none')
        translation_method: Name of translation pass ('unroller', 'translator', 'synthesis')
        scheduling_method: Name of scheduling pass.
            * ``'as_soon_as_possible'``: Schedule instructions greedily, as early as possible
            on a qubit resource. (alias: ``'asap'``)
            * ``'as_late_as_possible'``: Schedule instructions late, i.e. keeping qubits
            in the ground state when possible. (alias: ``'alap'``)
            If ``None``, no scheduling will be done.
        instruction_durations: Durations of instructions.
            The gate lengths defined in ``backend.properties`` are used as default.
            They are overwritten if this ``instruction_durations`` is specified.
            The format of ``instruction_durations`` must be as follows.
            The `instruction_durations` must be given as a list of tuples
            [(instruction_name, qubits, duration, unit), ...].
            | [('cx', [0, 1], 12.3, 'ns'), ('u3', [0], 4.56, 'ns')]
            | [('cx', [0, 1], 1000), ('u3', [0], 300)]
            If unit is omitted, the default is 'dt', which is a sample time depending on backend.
            If the time unit is 'dt', the duration must be an integer.
        dt: Backend sample time (resolution) in seconds.
            If ``None`` (default), ``backend.configuration().dt`` is used.
        approximation_degree (float): heuristic dial used for circuit approximation
            (1.0=no approximation, 0.0=maximal approximation)
        timing_constraints: An optional control hardware restriction on instruction time resolution.
            A quantum computer backend may report a set of restrictions, namely:

            - granularity: An integer value representing minimum pulse gate
              resolution in units of ``dt``. A user-defined pulse gate should have
              duration of a multiple of this granularity value.
            - min_length: An integer value representing minimum pulse gate
              length in units of ``dt``. A user-defined pulse gate should be longer
              than this length.
            - pulse_alignment: An integer value representing a time resolution of gate
              instruction starting time. Gate instruction should start at time which
              is a multiple of the alignment value.
            - acquire_alignment: An integer value representing a time resolution of measure
              instruction starting time. Measure instruction should start at time which
              is a multiple of the alignment value.

            This information will be provided by the backend configuration.
            If the backend doesn't have any restriction on the instruction time allocation,
            then ``timing_constraints`` is None and no adjustment will be performed.
        seed_transpiler: Sets random seed for the stochastic parts of the transpiler
        optimization_level: How much optimization to perform on the circuits.
            Higher levels generate more optimized circuits,
            at the expense of longer transpilation time.
            * 0: no optimization
            * 1: light optimization
            * 2: heavy optimization
            * 3: even heavier optimization
            If ``None``, level 1 will be chosen as default.
        pass_manager: The pass manager to use for a custom pipeline of transpiler passes.
            If this arg is present, all other args will be ignored and the
            pass manager will be used directly (Qiskit will not attempt to
            auto-select a pass manager based on transpile options).
        callback: A callback function that will be called after each
            pass execution. The function will be called with 5 keyword
            arguments,
            | ``pass_``: the pass being run.
            | ``dag``: the dag output of the pass.
            | ``time``: the time to execute the pass.
            | ``property_set``: the property set.
            | ``count``: the index for the pass execution.
            The exact arguments passed expose the internals of the pass manager,
            and are subject to change as the pass manager internals change. If
            you intend to reuse a callback function over multiple releases, be
            sure to check that the arguments being passed are the same.
            To use the callback feature, define a function that will
            take in kwargs dict and access the variables. For example::

                def callback_func(**kwargs):
                    pass_ = kwargs['pass_']
                    dag = kwargs['dag']
                    time = kwargs['time']
                    property_set = kwargs['property_set']
                    count = kwargs['count']
                    ...
                transpile(circ, callback=callback_func)

        output_name: A list with strings to identify the output circuits. The length of
            the list should be exactly the length of the ``circuits`` parameter.

    Returns:
        The transpiled circuit(s).

    Raises:
        TranspilerError: in case of bad inputs to transpiler (like conflicting parameters)
            or errors in passes
    """
    arg_circuits_list = isinstance(circuits, list)
    circuits = circuits if arg_circuits_list else [circuits]

    # transpiling schedules is not supported yet.
    start_time = time()
    if all(isinstance(c, Schedule) for c in circuits):
        warnings.warn("Transpiling schedules is not supported yet.", UserWarning)
        end_time = time()
        _log_transpile_time(start_time, end_time)
        if arg_circuits_list:
            return circuits
        else:
            return circuits[0]

    if pass_manager is not None:
        _check_conflicting_argument(
            optimization_level=optimization_level,
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            seed_transpiler=seed_transpiler,
            backend_properties=backend_properties,
            initial_layout=initial_layout,
            layout_method=layout_method,
            routing_method=routing_method,
            translation_method=translation_method,
            approximation_degree=approximation_degree,
            backend=backend,
        )

        warnings.warn(
            "The parameter pass_manager in transpile is being deprecated. "
            "The preferred way to tranpile a circuit using a custom pass manager is"
            " pass_manager.run(circuit)",
            DeprecationWarning,
            stacklevel=2,
        )
        return pass_manager.run(circuits, output_name=output_name, callback=callback)

    if optimization_level is None:
        # Take optimization level from the configuration or 1 as default.
        config = user_config.get_config()
        optimization_level = config.get("transpile_optimization_level", 1)

    if scheduling_method is not None and backend is None and not instruction_durations:
        warnings.warn(
            "When scheduling circuits without backend,"
            " 'instruction_durations' should be usually provided.",
            UserWarning,
        )

    # Get transpile_args to configure the circuit transpilation job(s)
    transpile_args = _parse_transpile_args(
        circuits,
        backend,
        basis_gates,
        coupling_map,
        backend_properties,
        initial_layout,
        layout_method,
        routing_method,
        translation_method,
        scheduling_method,
        instruction_durations,
        dt,
        approximation_degree,
        seed_transpiler,
        optimization_level,
        callback,
        output_name,
        timing_constraints,
    )

    _check_circuits_coupling_map(circuits, transpile_args, backend)

    # Transpile circuits in parallel
    circuits = parallel_map(_transpile_circuit, list(zip(circuits, transpile_args)))

    end_time = time()
    _log_transpile_time(start_time, end_time)

    if arg_circuits_list:
        return circuits
    else:
        return circuits[0]


def _check_conflicting_argument(**kargs):
    conflicting_args = [arg for arg, value in kargs.items() if value]
    if conflicting_args:
        raise TranspilerError(
            "The parameters pass_manager conflicts with the following "
            "parameter(s): {}.".format(", ".join(conflicting_args))
        )


def _check_circuits_coupling_map(circuits, transpile_args, backend):
    # Check circuit width against number of qubits in coupling_map(s)
    coupling_maps_list = list(
        config["pass_manager_config"].coupling_map for config in transpile_args
    )
    for circuit, parsed_coupling_map in zip(circuits, coupling_maps_list):
        # If coupling_map is not None or num_qubits == 1
        num_qubits = len(circuit.qubits)
        max_qubits = None
        if isinstance(parsed_coupling_map, CouplingMap):
            max_qubits = parsed_coupling_map.size()

        # If coupling_map is None, the limit might be in the backend (like in 1Q devices)
        elif backend is not None and not backend.configuration().simulator:
            max_qubits = backend.configuration().n_qubits

        if max_qubits is not None and (num_qubits > max_qubits):
            raise TranspilerError(
                f"Number of qubits ({num_qubits}) in {circuit.name} "
                f"is greater than maximum ({max_qubits}) in the coupling_map"
            )


def _log_transpile_time(start_time, end_time):
    log_msg = "Total Transpile Time - %.5f (ms)" % ((end_time - start_time) * 1000)
    logger.info(log_msg)


def _transpile_circuit(circuit_config_tuple: Tuple[QuantumCircuit, Dict]) -> QuantumCircuit:
    """Select a PassManager and run a single circuit through it.
    Args:
        circuit_config_tuple (tuple):
            circuit (QuantumCircuit): circuit to transpile
            transpile_config (dict): configuration dictating how to transpile. The
                dictionary has the following format:
                {'optimization_level': int,
                 'output_name': string,
                 'callback': callable,
                 'pass_manager_config': PassManagerConfig}
    Returns:
        The transpiled circuit
    Raises:
        TranspilerError: if transpile_config is not valid or transpilation incurs error
    """
    circuit, transpile_config = circuit_config_tuple

    pass_manager_config = transpile_config["pass_manager_config"]

    if transpile_config["faulty_qubits_map"]:
        pass_manager_config.initial_layout = _remap_layout_faulty_backend(
            pass_manager_config.initial_layout, transpile_config["faulty_qubits_map"]
        )

    # we choose an appropriate one based on desired optimization level
    level = transpile_config["optimization_level"]

    if level == 0:
        pass_manager = level_0_pass_manager(pass_manager_config)
    elif level == 1:
        pass_manager = level_1_pass_manager(pass_manager_config)
    elif level == 2:
        pass_manager = level_2_pass_manager(pass_manager_config)
    elif level == 3:
        pass_manager = level_3_pass_manager(pass_manager_config)
    else:
        raise TranspilerError("optimization_level can range from 0 to 3.")

    result = pass_manager.run(
        circuit, callback=transpile_config["callback"], output_name=transpile_config["output_name"]
    )

    if transpile_config["faulty_qubits_map"]:
        return _remap_circuit_faulty_backend(
            result,
            transpile_config["backend_num_qubits"],
            pass_manager_config.backend_properties,
            transpile_config["faulty_qubits_map"],
        )

    return result


def _remap_circuit_faulty_backend(circuit, num_qubits, backend_prop, faulty_qubits_map):
    faulty_qubits = backend_prop.faulty_qubits() if backend_prop else []
    disconnected_qubits = {k for k, v in faulty_qubits_map.items() if v is None}.difference(
        faulty_qubits
    )
    faulty_qubits_map_reverse = {v: k for k, v in faulty_qubits_map.items()}
    if faulty_qubits:
        faulty_qreg = circuit._create_qreg(len(faulty_qubits), "faulty")
    else:
        faulty_qreg = []
    if disconnected_qubits:
        disconnected_qreg = circuit._create_qreg(len(disconnected_qubits), "disconnected")
    else:
        disconnected_qreg = []

    new_layout = Layout()
    faulty_qubit = 0
    disconnected_qubit = 0

    for real_qubit in range(num_qubits):
        if faulty_qubits_map[real_qubit] is not None:
            new_layout[real_qubit] = circuit._layout[faulty_qubits_map[real_qubit]]
        else:
            if real_qubit in faulty_qubits:
                new_layout[real_qubit] = faulty_qreg[faulty_qubit]
                faulty_qubit += 1
            else:
                new_layout[real_qubit] = disconnected_qreg[disconnected_qubit]
                disconnected_qubit += 1
    physical_layout_dict = {}
    for index, qubit in enumerate(circuit.qubits):
        physical_layout_dict[qubit] = faulty_qubits_map_reverse[index]
    for qubit in faulty_qreg[:] + disconnected_qreg[:]:
        physical_layout_dict[qubit] = new_layout[qubit]
    dag_circuit = circuit_to_dag(circuit)
    apply_layout_pass = ApplyLayout()
    apply_layout_pass.property_set["layout"] = Layout(physical_layout_dict)
    circuit = dag_to_circuit(apply_layout_pass.run(dag_circuit))
    circuit._layout = new_layout
    return circuit


def _remap_layout_faulty_backend(layout, faulty_qubits_map):
    if layout is None:
        return layout
    new_layout = Layout()
    for virtual, physical in layout.get_virtual_bits().items():
        if faulty_qubits_map[physical] is None:
            raise TranspilerError(
                "The initial_layout parameter refers to faulty" " or disconnected qubits"
            )
        new_layout[virtual] = faulty_qubits_map[physical]
    return new_layout


def _parse_transpile_args(
    circuits,
    backend,
    basis_gates,
    coupling_map,
    backend_properties,
    initial_layout,
    layout_method,
    routing_method,
    translation_method,
    scheduling_method,
    instruction_durations,
    dt,
    approximation_degree,
    seed_transpiler,
    optimization_level,
    callback,
    output_name,
    timing_constraints,
) -> List[Dict]:
    """Resolve the various types of args allowed to the transpile() function through
    duck typing, overriding args, etc. Refer to the transpile() docstring for details on
    what types of inputs are allowed.

    Here the args are resolved by converting them to standard instances, and prioritizing
    them in case a transpile option is passed through multiple args (explicitly setting an
    arg has more priority than the arg set by backend).

    Returns:
        list[dicts]: a list of transpile parameters.

    Raises:
        TranspilerError: If instruction_durations are required but not supplied or found.
    """
    if initial_layout is not None and layout_method is not None:
        warnings.warn("initial_layout provided; layout_method is ignored.", UserWarning)
    # Each arg could be single or a list. If list, it must be the same size as
    # number of circuits. If single, duplicate to create a list of that size.
    num_circuits = len(circuits)

    basis_gates = _parse_basis_gates(basis_gates, backend, circuits)
    faulty_qubits_map = _parse_faulty_qubits_map(backend, num_circuits)
    coupling_map = _parse_coupling_map(coupling_map, backend, num_circuits)
    backend_properties = _parse_backend_properties(backend_properties, backend, num_circuits)
    backend_num_qubits = _parse_backend_num_qubits(backend, num_circuits)
    initial_layout = _parse_initial_layout(initial_layout, circuits)
    layout_method = _parse_layout_method(layout_method, num_circuits)
    routing_method = _parse_routing_method(routing_method, num_circuits)
    translation_method = _parse_translation_method(translation_method, num_circuits)
    approximation_degree = _parse_approximation_degree(approximation_degree, num_circuits)
    seed_transpiler = _parse_seed_transpiler(seed_transpiler, num_circuits)
    optimization_level = _parse_optimization_level(optimization_level, num_circuits)
    output_name = _parse_output_name(output_name, circuits)
    callback = _parse_callback(callback, num_circuits)
    durations = _parse_instruction_durations(backend, instruction_durations, dt, circuits)
    scheduling_method = _parse_scheduling_method(scheduling_method, num_circuits)
    timing_constraints = _parse_timing_constraints(backend, timing_constraints, num_circuits)
    if scheduling_method and any(d is None for d in durations):
        raise TranspilerError(
            "Transpiling a circuit with a scheduling method"
            "requires a backend or instruction_durations."
        )

    list_transpile_args = []
    for args in zip(
        basis_gates,
        coupling_map,
        backend_properties,
        initial_layout,
        layout_method,
        routing_method,
        translation_method,
        scheduling_method,
        durations,
        approximation_degree,
        timing_constraints,
        seed_transpiler,
        optimization_level,
        output_name,
        callback,
        backend_num_qubits,
        faulty_qubits_map,
    ):
        transpile_args = {
            "pass_manager_config": PassManagerConfig(
                basis_gates=args[0],
                coupling_map=args[1],
                backend_properties=args[2],
                initial_layout=args[3],
                layout_method=args[4],
                routing_method=args[5],
                translation_method=args[6],
                scheduling_method=args[7],
                instruction_durations=args[8],
                approximation_degree=args[9],
                timing_constraints=args[10],
                seed_transpiler=args[11],
            ),
            "optimization_level": args[12],
            "output_name": args[13],
            "callback": args[14],
            "backend_num_qubits": args[15],
            "faulty_qubits_map": args[16],
        }
        list_transpile_args.append(transpile_args)

    return list_transpile_args


def _create_faulty_qubits_map(backend):
    """If the backend has faulty qubits, those should be excluded. A faulty_qubit_map is a map
    from working qubit in the backend to dummy qubits that are consecutive and connected."""
    faulty_qubits_map = None
    if backend is not None:
        if backend.properties():
            faulty_qubits = backend.properties().faulty_qubits()
            faulty_edges = [gates.qubits for gates in backend.properties().faulty_gates()]
        else:
            faulty_qubits = []
            faulty_edges = []

        if faulty_qubits or faulty_edges:
            faulty_qubits_map = {}
            configuration = backend.configuration()
            full_coupling_map = configuration.coupling_map
            functional_cm_list = [
                edge
                for edge in full_coupling_map
                if (set(edge).isdisjoint(faulty_qubits) and edge not in faulty_edges)
            ]

            connected_working_qubits = CouplingMap(functional_cm_list).largest_connected_component()
            dummy_qubit_counter = 0
            for qubit in range(configuration.n_qubits):
                if qubit in connected_working_qubits:
                    faulty_qubits_map[qubit] = dummy_qubit_counter
                    dummy_qubit_counter += 1
                else:
                    faulty_qubits_map[qubit] = None
    return faulty_qubits_map


def _parse_basis_gates(basis_gates, backend, circuits):
    # try getting basis_gates from user, else backend
    if basis_gates is None:
        if getattr(backend, "configuration", None):
            basis_gates = getattr(backend.configuration(), "basis_gates", None)
    # basis_gates could be None, or a list of basis, e.g. ['u3', 'cx']
    if basis_gates is None or (
        isinstance(basis_gates, list) and all(isinstance(i, str) for i in basis_gates)
    ):
        basis_gates = [basis_gates] * len(circuits)

    return basis_gates


def _parse_coupling_map(coupling_map, backend, num_circuits):
    # try getting coupling_map from user, else backend
    if coupling_map is None:
        if getattr(backend, "configuration", None):
            configuration = backend.configuration()
            if hasattr(configuration, "coupling_map") and configuration.coupling_map:
                faulty_map = _create_faulty_qubits_map(backend)
                if faulty_map:
                    faulty_edges = [gate.qubits for gate in backend.properties().faulty_gates()]
                    functional_gates = [
                        edge for edge in configuration.coupling_map if edge not in faulty_edges
                    ]
                    coupling_map = CouplingMap()
                    for qubit1, qubit2 in functional_gates:
                        if faulty_map[qubit1] is not None and faulty_map[qubit2] is not None:
                            coupling_map.add_edge(faulty_map[qubit1], faulty_map[qubit2])
                    if configuration.n_qubits != coupling_map.size():
                        warnings.warn(
                            "The backend has currently some qubits/edges out of service."
                            " This temporarily reduces the backend size from "
                            f"{configuration.n_qubits} to {coupling_map.size()}",
                            UserWarning,
                        )
                else:
                    coupling_map = CouplingMap(configuration.coupling_map)

    # coupling_map could be None, or a list of lists, e.g. [[0, 1], [2, 1]]
    if coupling_map is None or isinstance(coupling_map, CouplingMap):
        coupling_map = [coupling_map] * num_circuits
    elif isinstance(coupling_map, list) and all(
        isinstance(i, list) and len(i) == 2 for i in coupling_map
    ):
        coupling_map = [coupling_map] * num_circuits

    coupling_map = [CouplingMap(cm) if isinstance(cm, list) else cm for cm in coupling_map]

    return coupling_map


def _parse_backend_properties(backend_properties, backend, num_circuits):
    # try getting backend_properties from user, else backend
    if backend_properties is None:
        if getattr(backend, "properties", None):
            backend_properties = backend.properties()
            if backend_properties and (
                backend_properties.faulty_qubits() or backend_properties.faulty_gates()
            ):
                faulty_qubits = sorted(backend_properties.faulty_qubits(), reverse=True)
                faulty_edges = [gates.qubits for gates in backend_properties.faulty_gates()]
                # remove faulty qubits in backend_properties.qubits
                for faulty_qubit in faulty_qubits:
                    del backend_properties.qubits[faulty_qubit]

                gates = []
                for gate in backend_properties.gates:
                    # remove gates using faulty edges or with faulty qubits (and remap the
                    # gates in terms of faulty_qubits_map)
                    faulty_qubits_map = _create_faulty_qubits_map(backend)
                    if (
                        any(faulty_qubits_map[qubits] is not None for qubits in gate.qubits)
                        or gate.qubits in faulty_edges
                    ):
                        continue
                    gate_dict = gate.to_dict()
                    replacement_gate = Gate.from_dict(gate_dict)
                    gate_dict["qubits"] = [faulty_qubits_map[qubit] for qubit in gate.qubits]
                    args = "_".join([str(qubit) for qubit in gate_dict["qubits"]])
                    gate_dict["name"] = "{}{}".format(gate_dict["gate"], args)
                    gates.append(replacement_gate)

                backend_properties.gates = gates
    if not isinstance(backend_properties, list):
        backend_properties = [backend_properties] * num_circuits
    return backend_properties


def _parse_backend_num_qubits(backend, num_circuits):
    if backend is None:
        return [None] * num_circuits
    if not isinstance(backend, list):
        return [backend.configuration().n_qubits] * num_circuits
    backend_num_qubits = []
    for a_backend in backend:
        backend_num_qubits.append(a_backend.configuration().n_qubits)
    return backend_num_qubits


def _parse_initial_layout(initial_layout, circuits):
    # initial_layout could be None, or a list of ints, e.g. [0, 5, 14]
    # or a list of tuples/None e.g. [qr[0], None, qr[1]] or a dict e.g. {qr[0]: 0}
    def _layout_from_raw(initial_layout, circuit):
        if initial_layout is None or isinstance(initial_layout, Layout):
            return initial_layout
        elif isinstancelist(initial_layout):
            if all(isinstanceint(elem) for elem in initial_layout):
                initial_layout = Layout.from_intlist(initial_layout, *circuit.qregs)
            elif all(elem is None or isinstance(elem, Qubit) for elem in initial_layout):
                initial_layout = Layout.from_qubit_list(initial_layout, *circuit.qregs)
        elif isinstance(initial_layout, dict):
            initial_layout = Layout(initial_layout)
        else:
            raise TranspilerError("The initial_layout parameter could not be parsed")
        return initial_layout

    # multiple layouts?
    if isinstance(initial_layout, list) and any(
        isinstance(i, (list, dict)) for i in initial_layout
    ):
        initial_layout = [
            _layout_from_raw(lo, circ) if isinstance(lo, (list, dict)) else lo
            for lo, circ in zip(initial_layout, circuits)
        ]
    else:
        # even if one layout, but multiple circuits, the layout needs to be adapted for each
        initial_layout = [_layout_from_raw(initial_layout, circ) for circ in circuits]

    if not isinstance(initial_layout, list):
        initial_layout = [initial_layout] * len(circuits)

    return initial_layout


def _parse_layout_method(layout_method, num_circuits):
    if not isinstance(layout_method, list):
        layout_method = [layout_method] * num_circuits
    return layout_method


def _parse_routing_method(routing_method, num_circuits):
    if not isinstance(routing_method, list):
        routing_method = [routing_method] * num_circuits
    return routing_method


def _parse_translation_method(translation_method, num_circuits):
    if not isinstance(translation_method, list):
        translation_method = [translation_method] * num_circuits
    return translation_method


def _parse_scheduling_method(scheduling_method, num_circuits):
    if not isinstance(scheduling_method, list):
        scheduling_method = [scheduling_method] * num_circuits
    return scheduling_method


def _parse_instruction_durations(backend, inst_durations, dt, circuits):
    """Create a list of ``InstructionDuration``s. If ``inst_durations`` is provided,
    the backend will be ignored, otherwise, the durations will be populated from the
    backend. If any circuits have gate calibrations, those calibration durations would
    take precedence over backend durations, but be superceded by ``inst_duration``s.
    """
    if not inst_durations:
        backend_durations = InstructionDurations()
        try:
            backend_durations = InstructionDurations.from_backend(backend)
        except AttributeError:
            pass

    durations = []
    for circ in circuits:
        circ_durations = InstructionDurations()
        if not inst_durations:
            circ_durations.update(backend_durations, dt or backend_durations.dt)

        if circ.calibrations:
            cal_durations = []
            for gate, gate_cals in circ.calibrations.items():
                for (qubits, _), schedule in gate_cals.items():
                    cal_durations.append((gate, qubits, schedule.duration))
            circ_durations.update(cal_durations, circ_durations.dt)

        if inst_durations:
            circ_durations.update(inst_durations, dt or getattr(inst_durations, "dt", None))

        durations.append(circ_durations)
    return durations


def _parse_approximation_degree(approximation_degree, num_circuits):
    if not isinstance(approximation_degree, list):
        approximation_degree = [approximation_degree] * num_circuits
    if not all(0.0 <= d <= 1.0 for d in approximation_degree if d):
        raise TranspilerError("Approximation degree must be in [0.0, 1.0]")
    return approximation_degree


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


def _parse_callback(callback, num_circuits):
    if not isinstance(callback, list):
        callback = [callback] * num_circuits
    return callback


def _parse_faulty_qubits_map(backend, num_circuits):
    if backend is None:
        return [None] * num_circuits
    if not isinstance(backend, list):
        return [_create_faulty_qubits_map(backend)] * num_circuits
    faulty_qubits_map = []
    for a_backend in backend:
        faulty_qubits_map.append(_create_faulty_qubits_map(a_backend))
    return faulty_qubits_map


def _parse_output_name(output_name, circuits):
    # naming and returning circuits
    # output_name could be either a string or a list
    if output_name is not None:
        if isinstance(output_name, str):
            # single circuit
            if len(circuits) == 1:
                return [output_name]
            # multiple circuits
            else:
                raise TranspilerError(
                    "Expected a list object of length equal "
                    + "to that of the number of circuits "
                    + "being transpiled"
                )
        elif isinstance(output_name, list):
            if len(circuits) == len(output_name) and all(
                isinstance(name, str) for name in output_name
            ):
                return output_name
            else:
                raise TranspilerError(
                    "The length of output_name list "
                    "must be equal to the number of "
                    "transpiled circuits and the output_name "
                    "list should be strings."
                )
        else:
            raise TranspilerError(
                "The parameter output_name should be a string or a"
                "list of strings: %s was used." % type(output_name)
            )
    else:
        return [circuit.name for circuit in circuits]


def _parse_timing_constraints(backend, timing_constraints, num_circuits):

    if backend is None and timing_constraints is None:
        timing_constraints = TimingConstraints()
    else:
        if timing_constraints is None:
            # get constraints from backend
            timing_constraints = getattr(backend.configuration(), "timing_constraints", {})
        timing_constraints = TimingConstraints(**timing_constraints)

    return [timing_constraints] * num_circuits
