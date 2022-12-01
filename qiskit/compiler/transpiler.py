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

# pylint: disable=invalid-sequence-index

"""Circuit transpile function"""
import io
from itertools import cycle
import logging
import os
import pickle
import sys
from time import time
from typing import List, Union, Dict, Callable, Any, Optional, Tuple, Iterable
import warnings

from qiskit import user_config
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import Qubit
from qiskit.converters import isinstanceint, isinstancelist, dag_to_circuit, circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.providers.backend import Backend
from qiskit.providers.models import BackendProperties
from qiskit.providers.models.backendproperties import Gate
from qiskit.pulse import Schedule, InstructionScheduleMap
from qiskit.tools import parallel
from qiskit.transpiler import Layout, CouplingMap, PropertySet
from qiskit.transpiler.basepasses import BasePass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations, InstructionDurationsType
from qiskit.transpiler.passes import ApplyLayout
from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.preset_passmanagers import (
    level_0_pass_manager,
    level_1_pass_manager,
    level_2_pass_manager,
    level_3_pass_manager,
)
from qiskit.transpiler.timing_constraints import TimingConstraints
from qiskit.transpiler.target import Target, target_to_backend_properties

if sys.version_info >= (3, 8):
    from multiprocessing.shared_memory import SharedMemory  # pylint: disable=no-name-in-module
    from multiprocessing.managers import SharedMemoryManager  # pylint: disable=no-name-in-module
else:
    from shared_memory import SharedMemory, SharedMemoryManager

logger = logging.getLogger(__name__)


def transpile(
    circuits: Union[QuantumCircuit, List[QuantumCircuit]],
    backend: Optional[Backend] = None,
    basis_gates: Optional[List[str]] = None,
    inst_map: Optional[List[InstructionScheduleMap]] = None,
    coupling_map: Optional[Union[CouplingMap, List[List[int]]]] = None,
    backend_properties: Optional[BackendProperties] = None,
    initial_layout: Optional[Union[Layout, Dict, List]] = None,
    layout_method: Optional[str] = None,
    routing_method: Optional[str] = None,
    translation_method: Optional[str] = None,
    scheduling_method: Optional[str] = None,
    instruction_durations: Optional[InstructionDurationsType] = None,
    dt: Optional[float] = None,
    approximation_degree: Optional[float] = 1.0,
    timing_constraints: Optional[Dict[str, int]] = None,
    seed_transpiler: Optional[int] = None,
    optimization_level: Optional[int] = None,
    callback: Optional[Callable[[BasePass, DAGCircuit, float, PropertySet, int], Any]] = None,
    output_name: Optional[Union[str, List[str]]] = None,
    unitary_synthesis_method: str = "default",
    unitary_synthesis_plugin_config: dict = None,
    target: Target = None,
    hls_config: Optional[HLSConfig] = None,
    init_method: str = None,
    optimization_method: str = None,
    ignore_backend_supplied_default_methods: bool = False,
) -> Union[QuantumCircuit, List[QuantumCircuit]]:
    """Transpile one or more circuits, according to some desired transpilation targets.

    .. deprecated:: 0.23.0

        Previously, all arguments accepted lists of the same length as ``circuits``,
        which was used to specialize arguments for circuits at the corresponding
        indices. Support for using such argument lists is now deprecated and will
        be removed in the 0.25.0 release. If you need to use multiple values for an
        argument, you can use multiple :func:`~.transpile` calls (and potentially
        :func:`~.parallel_map` to leverage multiprocessing if needed).

    Transpilation is done in parallel using multiprocessing.

    Args:
        circuits: Circuit(s) to transpile
        backend: If set, the transpiler will compile the input circuit to this target
            device. If any other option is explicitly set (e.g., ``coupling_map``), it
            will override the backend's.
        basis_gates: List of basis gate names to unroll to
            (e.g: ``['u1', 'u2', 'u3', 'cx']``). If ``None``, do not unroll.
        inst_map: Mapping of unrolled gates to pulse schedules. If this is not provided,
            transpiler tries to get from the backend. If any user defined calibration
            is found in the map and this is used in a circuit, transpiler attaches
            the custom gate definition to the circuit. This enables one to flexibly
            override the low-level instruction implementation. This feature is available
            iff the backend supports the pulse gate experiment.
        coupling_map: Directed coupling map (perhaps custom) to target in mapping. If
            the coupling map is symmetric, both directions need to be specified.

            Multiple formats are supported:

            #. ``CouplingMap`` instance
            #. List, must be given as an adjacency matrix, where each entry
               specifies all directed two-qubit interactions supported by backend,
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

        layout_method: Name of layout selection pass ('trivial', 'dense', 'noise_adaptive', 'sabre').
            This can also be the external plugin name to use for the ``layout`` stage.
            You can see a list of installed plugins by using :func:`~.list_stage_plugins` with
            ``"layout"`` for the ``stage_name`` argument.
        routing_method: Name of routing pass
            ('basic', 'lookahead', 'stochastic', 'sabre', 'none'). Note
            This can also be the external plugin name to use for the ``routing`` stage.
            You can see a list of installed plugins by using :func:`~.list_stage_plugins` with
            ``"routing"`` for the ``stage_name`` argument.
        translation_method: Name of translation pass ('unroller', 'translator', 'synthesis')
            This can also be the external plugin name to use for the ``translation`` stage.
            You can see a list of installed plugins by using :func:`~.list_stage_plugins` with
            ``"translation"`` for the ``stage_name`` argument.
        scheduling_method: Name of scheduling pass.
            * ``'as_soon_as_possible'``: Schedule instructions greedily, as early as possible
            on a qubit resource. (alias: ``'asap'``)
            * ``'as_late_as_possible'``: Schedule instructions late, i.e. keeping qubits
            in the ground state when possible. (alias: ``'alap'``)
            If ``None``, no scheduling will be done. This can also be the external plugin name
            to use for the ``scheduling`` stage. You can see a list of installed plugins by
            using :func:`~.list_stage_plugins` with ``"scheduling"`` for the ``stage_name``
            argument.
        instruction_durations: Durations of instructions.
            Applicable only if scheduling_method is specified.
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
        unitary_synthesis_method (str): The name of the unitary synthesis
            method to use. By default 'default' is used, which is the only
            method included with qiskit. If you have installed any unitary
            synthesis plugins you can use the name exported by the plugin.
        unitary_synthesis_plugin_config: An optional configuration dictionary
            that will be passed directly to the unitary synthesis plugin. By
            default this setting will have no effect as the default unitary
            synthesis method does not take custom configuration. This should
            only be necessary when a unitary synthesis plugin is specified with
            the ``unitary_synthesis`` argument. As this is custom for each
            unitary synthesis plugin refer to the plugin documentation for how
            to use this option.
        target: A backend transpiler target. Normally this is specified as part of
            the ``backend`` argument, but if you have manually constructed a
            :class:`~qiskit.transpiler.Target` object you can specify it manually here.
            This will override the target from ``backend``.
        hls_config: An optional configuration class
            :class:`~qiskit.transpiler.passes.synthesis.HLSConfig` that will be passed directly
            to :class:`~qiskit.transpiler.passes.synthesis.HighLevelSynthesis` transformation pass.
            This configuration class allows to specify for various high-level objects the lists of
            synthesis algorithms and their parameters.
        init_method: The plugin name to use for the ``init`` stage. By default an external
            plugin is not used. You can see a list of installed plugins by
            using :func:`~.list_stage_plugins` with ``"init"`` for the stage
            name argument.
        optimization_method: The plugin name to use for the
            ``optimization`` stage. By default an external
            plugin is not used. You can see a list of installed plugins by
            using :func:`~.list_stage_plugins` with ``"optimization"`` for the
            ``stage_name`` argument.
        ignore_backend_supplied_default_methods: If set to ``True`` any default methods specified by
            a backend will be ignored. Some backends specify alternative default methods
            to support custom compilation target-specific passes/plugins which support
            backend-specific compilation techniques. If you'd prefer that these defaults were
            not used this option is used to disable those backend-specific defaults.

    Returns:
        The transpiled circuit(s).

    Raises:
        TranspilerError: in case of bad inputs to transpiler (like conflicting parameters)
            or errors in passes
    """
    arg_circuits_list = isinstance(circuits, list)
    circuits = circuits if arg_circuits_list else [circuits]

    if not circuits:
        return []

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

    if optimization_level is None:
        # Take optimization level from the configuration or 1 as default.
        config = user_config.get_config()
        optimization_level = config.get("transpile_optimization_level", 1)

    if (
        scheduling_method is not None
        and backend is None
        and target is None
        and not instruction_durations
    ):
        warnings.warn(
            "When scheduling circuits without backend,"
            " 'instruction_durations' should be usually provided.",
            UserWarning,
        )

    unique_transpile_args, shared_args = _parse_transpile_args(
        circuits,
        backend,
        basis_gates,
        inst_map,
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
        unitary_synthesis_method,
        unitary_synthesis_plugin_config,
        target,
        hls_config,
        init_method,
        optimization_method,
        ignore_backend_supplied_default_methods,
    )
    # Get transpile_args to configure the circuit transpilation job(s)
    if coupling_map in unique_transpile_args:
        cmap_conf = unique_transpile_args["coupling_map"]
    else:
        cmap_conf = [shared_args["coupling_map"]] * len(circuits)
    _check_circuits_coupling_map(circuits, cmap_conf, backend)
    if (
        len(circuits) > 1
        and os.getenv("QISKIT_IN_PARALLEL", "FALSE") == "FALSE"
        and parallel.PARALLEL_DEFAULT
    ):
        with SharedMemoryManager() as smm:
            with io.BytesIO() as buf:
                pickle.dump(shared_args, buf)
                data = buf.getvalue()
            smb = smm.SharedMemory(size=len(data))
            smb.buf[: len(data)] = data[:]
            # Transpile circuits in parallel
            circuits = parallel.parallel_map(
                _transpile_circuit,
                list(zip(circuits, cycle([smb.name]), unique_transpile_args)),
            )
    else:
        output_circuits = []
        for circuit, unique_args in zip(circuits, unique_transpile_args):
            transpile_config, pass_manager = _combine_args(shared_args, unique_args)
            output_circuits.append(
                _serial_transpile_circuit(
                    circuit,
                    pass_manager,
                    transpile_config["callback"],
                    transpile_config["output_name"],
                    transpile_config["backend_num_qubits"],
                    transpile_config["faulty_qubits_map"],
                    transpile_config["pass_manager_config"].backend_properties,
                )
            )
        circuits = output_circuits
    end_time = time()
    _log_transpile_time(start_time, end_time)

    if arg_circuits_list:
        return circuits
    else:
        return circuits[0]


def _check_circuits_coupling_map(circuits, cmap_conf, backend):
    # Check circuit width against number of qubits in coupling_map(s)
    coupling_maps_list = cmap_conf
    for circuit, parsed_coupling_map in zip(circuits, coupling_maps_list):
        # If coupling_map is not None or num_qubits == 1
        num_qubits = len(circuit.qubits)
        max_qubits = None
        if isinstance(parsed_coupling_map, CouplingMap):
            max_qubits = parsed_coupling_map.size()

        # If coupling_map is None, the limit might be in the backend (like in 1Q devices)
        elif backend is not None:
            backend_version = getattr(backend, "version", 0)
            if backend_version <= 1:
                if not backend.configuration().simulator:
                    max_qubits = backend.configuration().n_qubits
            else:
                max_qubits = backend.num_qubits

        if max_qubits is not None and (num_qubits > max_qubits):
            raise TranspilerError(
                f"Number of qubits ({num_qubits}) in {circuit.name} "
                f"is greater than maximum ({max_qubits}) in the coupling_map"
            )


def _log_transpile_time(start_time, end_time):
    log_msg = "Total Transpile Time - %.5f (ms)" % ((end_time - start_time) * 1000)
    logger.info(log_msg)


def _combine_args(shared_transpiler_args, unique_config):
    # Pop optimization_level to exclude it from the kwargs when building a
    # PassManagerConfig
    level = shared_transpiler_args.pop("optimization_level")
    pass_manager_config = shared_transpiler_args
    pass_manager_config.update(unique_config.pop("pass_manager_config"))
    pass_manager_config = PassManagerConfig(**pass_manager_config)
    # restore optimization_level in the input shared dict in case it's used again
    # in the same process
    shared_transpiler_args["optimization_level"] = level

    transpile_config = unique_config
    transpile_config["pass_manager_config"] = pass_manager_config

    if transpile_config["faulty_qubits_map"]:
        pass_manager_config.initial_layout = _remap_layout_faulty_backend(
            pass_manager_config.initial_layout, transpile_config["faulty_qubits_map"]
        )

    # we choose an appropriate one based on desired optimization level
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
    return transpile_config, pass_manager


def _serial_transpile_circuit(
    circuit,
    pass_manager,
    callback,
    output_name,
    num_qubits,
    faulty_qubits_map=None,
    backend_prop=None,
):
    result = pass_manager.run(circuit, callback=callback, output_name=output_name)
    if faulty_qubits_map:
        return _remap_circuit_faulty_backend(
            result,
            num_qubits,
            backend_prop,
            faulty_qubits_map,
        )

    return result


def _transpile_circuit(circuit_config_tuple: Tuple[QuantumCircuit, str, Dict]) -> QuantumCircuit:
    """Select a PassManager and run a single circuit through it.
    Args:
        circuit_config_tuple (tuple):
            circuit (QuantumCircuit): circuit to transpile
            name (str): The name of the shared memory object containing a pickled dict of shared
                arguments between parallel works
            unique_config (dict): configuration dictating unique arguments for transpile.
    Returns:
        The transpiled circuit
    Raises:
        TranspilerError: if transpile_config is not valid or transpilation incurs error
    """
    circuit, name, unique_config = circuit_config_tuple
    existing_shm = SharedMemory(name=name)
    try:
        with io.BytesIO(existing_shm.buf) as buf:
            shared_transpiler_args = pickle.load(buf)
    finally:
        existing_shm.close()

    transpile_config, pass_manager = _combine_args(shared_transpiler_args, unique_config)
    pass_manager_config = transpile_config["pass_manager_config"]

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
            new_layout[real_qubit] = circuit._layout.initial_layout[faulty_qubits_map[real_qubit]]
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
                "The initial_layout parameter refers to faulty or disconnected qubits"
            )
        new_layout[virtual] = faulty_qubits_map[physical]
    return new_layout


def _parse_transpile_args(
    circuits,
    backend,
    basis_gates,
    inst_map,
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
    unitary_synthesis_method,
    unitary_synthesis_plugin_config,
    target,
    hls_config,
    init_method,
    optimization_method,
    ignore_backend_supplied_default_methods,
) -> Tuple[List[Dict], Dict]:
    """Resolve the various types of args allowed to the transpile() function through
    duck typing, overriding args, etc. Refer to the transpile() docstring for details on
    what types of inputs are allowed.

    Here the args are resolved by converting them to standard instances, and prioritizing
    them in case a transpile option is passed through multiple args (explicitly setting an
    arg has more priority than the arg set by backend).

    Returns:
        Tuple[list[dict], dict]: a tuple contain a list of unique transpile parameter dicts and
        the second element contains a dict of shared transpiler argument across all circuits.

    Raises:
        TranspilerError: If instruction_durations are required but not supplied or found.
    """
    if initial_layout is not None and layout_method is not None:
        warnings.warn("initial_layout provided; layout_method is ignored.", UserWarning)
    # Each arg could be single or a list. If list, it must be the same size as
    # number of circuits. If single, duplicate to create a list of that size.
    num_circuits = len(circuits)
    user_input_durations = instruction_durations
    user_input_timing_constraints = timing_constraints
    user_input_initial_layout = initial_layout
    # If a target is specified have it override any implicit selections from a backend
    # but if an argument is explicitly passed use that instead of the target version
    if target is not None:
        if coupling_map is None:
            coupling_map = target.build_coupling_map()
        if basis_gates is None:
            basis_gates = list(target.operation_names)
        if instruction_durations is None:
            instruction_durations = target.durations()
        if inst_map is None:
            inst_map = target.instruction_schedule_map()
        if dt is None:
            dt = target.dt
        if timing_constraints is None:
            timing_constraints = target.timing_constraints()
        if backend_properties is None:
            backend_properties = target_to_backend_properties(target)

    basis_gates = _parse_basis_gates(basis_gates, backend)
    initial_layout = _parse_initial_layout(initial_layout, circuits)
    inst_map = _parse_inst_map(inst_map, backend)
    faulty_qubits_map = _parse_faulty_qubits_map(backend, num_circuits)
    coupling_map = _parse_coupling_map(coupling_map, backend)
    backend_properties = _parse_backend_properties(backend_properties, backend)
    backend_num_qubits = _parse_backend_num_qubits(backend, num_circuits)
    approximation_degree = _parse_approximation_degree(approximation_degree)
    output_name = _parse_output_name(output_name, circuits)
    callback = _parse_callback(callback, num_circuits)
    durations = _parse_instruction_durations(backend, instruction_durations, dt, circuits)
    timing_constraints = _parse_timing_constraints(backend, timing_constraints, num_circuits)
    target = _parse_target(backend, target)
    if scheduling_method and any(d is None for d in durations):
        raise TranspilerError(
            "Transpiling a circuit with a scheduling method"
            "requires a backend or instruction_durations."
        )
    unique_dict = {
        "callback": callback,
        "output_name": output_name,
        "faulty_qubits_map": faulty_qubits_map,
        "backend_num_qubits": backend_num_qubits,
    }
    shared_dict = {
        "optimization_level": optimization_level,
        "basis_gates": basis_gates,
        "init_method": init_method,
        "optimization_method": optimization_method,
    }

    list_transpile_args = []
    if not ignore_backend_supplied_default_methods:
        if scheduling_method is None and hasattr(backend, "get_scheduling_stage_plugin"):
            scheduling_method = backend.get_scheduling_stage_plugin()
        if translation_method is None and hasattr(backend, "get_translation_stage_plugin"):
            translation_method = backend.get_translation_stage_plugin()

    for key, value in {
        "inst_map": inst_map,
        "coupling_map": coupling_map,
        "backend_properties": backend_properties,
        "approximation_degree": approximation_degree,
        "initial_layout": initial_layout,
        "layout_method": layout_method,
        "routing_method": routing_method,
        "translation_method": translation_method,
        "scheduling_method": scheduling_method,
        "instruction_durations": durations,
        "timing_constraints": timing_constraints,
        "seed_transpiler": seed_transpiler,
        "unitary_synthesis_method": unitary_synthesis_method,
        "unitary_synthesis_plugin_config": unitary_synthesis_plugin_config,
        "target": target,
        "hls_config": hls_config,
    }.items():
        if isinstance(value, list):
            # This giant if-statement detects deprecated use of argument
            # broadcasting. For arguments that previously supported broadcast
            # but were not themselves of type list (the majority), we simply warn
            # when the user provides a list. For the others, special handling is
            # required to disambiguate an expected value of type list from
            # an attempt to provide multiple values for broadcast. This path is
            # super buggy in general (outside of the warning) and since we're
            # deprecating this it's better to just remove it than try to clean it up.
            # pylint: disable=too-many-boolean-expressions
            if (
                key not in {"instruction_durations", "timing_constraints", "initial_layout"}
                or (
                    key == "initial_layout"
                    and user_input_initial_layout
                    and isinstance(user_input_initial_layout, list)
                    and isinstance(user_input_initial_layout[0], (Layout, dict, list))
                )
                or (
                    key == "instruction_durations"
                    and user_input_durations
                    and isinstance(user_input_durations, list)
                    and isinstance(user_input_durations[0], (list, InstructionDurations))
                )
                or (
                    key == "timing_constraints"
                    and user_input_timing_constraints
                    and isinstance(user_input_timing_constraints, list)
                )
            ):
                warnings.warn(
                    f"Passing in a list of arguments for {key} is deprecated and will no longer work "
                    "starting in the 0.25.0 release.",
                    DeprecationWarning,
                    stacklevel=3,
                )
            unique_dict[key] = value
        else:
            shared_dict[key] = value

    for kwargs in _zip_dict(unique_dict):
        transpile_args = {
            "output_name": kwargs.pop("output_name"),
            "callback": kwargs.pop("callback"),
            "faulty_qubits_map": kwargs.pop("faulty_qubits_map"),
            "backend_num_qubits": kwargs.pop("backend_num_qubits"),
            "pass_manager_config": kwargs,
        }
        list_transpile_args.append(transpile_args)

    return list_transpile_args, shared_dict


def _create_faulty_qubits_map(backend):
    """If the backend has faulty qubits, those should be excluded. A faulty_qubit_map is a map
    from working qubit in the backend to dummy qubits that are consecutive and connected."""
    faulty_qubits_map = None
    if backend is not None:
        backend_version = getattr(backend, "version", 0)
        if backend_version > 1:
            return None
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


def _parse_basis_gates(basis_gates, backend):
    # try getting basis_gates from user, else backend
    if basis_gates is None:
        backend_version = getattr(backend, "version", 0)
        if backend_version <= 1:
            if getattr(backend, "configuration", None):
                basis_gates = getattr(backend.configuration(), "basis_gates", None)
        else:
            basis_gates = backend.operation_names
    return basis_gates


def _parse_inst_map(inst_map, backend):
    # try getting inst_map from user, else backend
    if inst_map is None:
        backend_version = getattr(backend, "version", 0)
        if backend_version <= 1:
            if hasattr(backend, "defaults"):
                inst_map = getattr(backend.defaults(), "instruction_schedule_map", None)
        else:
            inst_map = backend.target.instruction_schedule_map()
    return inst_map


def _parse_coupling_map(coupling_map, backend):
    # try getting coupling_map from user, else backend
    if coupling_map is None:
        backend_version = getattr(backend, "version", 0)
        if backend_version <= 1:
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
        else:
            coupling_map = backend.coupling_map

    # coupling_map could be None, or a list of lists, e.g. [[0, 1], [2, 1]]
    if coupling_map is None or isinstance(coupling_map, CouplingMap):
        return coupling_map
    if isinstance(coupling_map, list) and all(
        isinstance(i, list) and len(i) == 2 for i in coupling_map
    ):
        return CouplingMap(coupling_map)

    coupling_map = [CouplingMap(cm) if isinstance(cm, list) else cm for cm in coupling_map]
    return coupling_map


def _parse_backend_properties(backend_properties, backend):
    # try getting backend_properties from user, else backend
    if backend_properties is None:
        backend_version = getattr(backend, "version", 0)
        if backend_version <= 1:
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
        else:
            backend_properties = target_to_backend_properties(backend.target)
    return backend_properties


def _parse_backend_num_qubits(backend, num_circuits):
    if backend is None:
        return [None] * num_circuits
    if not isinstance(backend, list):
        backend_version = getattr(backend, "version", 0)
        if backend_version <= 1:
            return [backend.configuration().n_qubits] * num_circuits
        else:
            return [backend.num_qubits] * num_circuits
    backend_num_qubits = []
    for a_backend in backend:
        backend_version = getattr(backend, "version", 0)
        if backend_version <= 1:
            backend_num_qubits.append(a_backend.configuration().n_qubits)
        else:
            backend_num_qubits.append(a_backend.num_qubits)
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

    return initial_layout


def _parse_instruction_durations(backend, inst_durations, dt, circuits):
    """Create a list of ``InstructionDuration``s. If ``inst_durations`` is provided,
    the backend will be ignored, otherwise, the durations will be populated from the
    backend. If any circuits have gate calibrations, those calibration durations would
    take precedence over backend durations, but be superceded by ``inst_duration``s.
    """
    if not inst_durations:
        backend_version = getattr(backend, "version", 0)
        if backend_version <= 1:
            backend_durations = InstructionDurations()
            try:
                backend_durations = InstructionDurations.from_backend(backend)
            except AttributeError:
                pass
        else:
            backend_durations = backend.instruction_durations

    durations = []
    for circ in circuits:
        circ_durations = InstructionDurations()
        if not inst_durations:
            circ_durations.update(backend_durations, dt or backend_durations.dt)

        if circ.calibrations:
            cal_durations = []
            for gate, gate_cals in circ.calibrations.items():
                for (qubits, parameters), schedule in gate_cals.items():
                    cal_durations.append((gate, qubits, parameters, schedule.duration))
            circ_durations.update(cal_durations, circ_durations.dt)

        if inst_durations:
            circ_durations.update(inst_durations, dt or getattr(inst_durations, "dt", None))

        durations.append(circ_durations)
    return durations


def _parse_approximation_degree(approximation_degree):
    if approximation_degree is None:
        return None
    if not isinstance(approximation_degree, list):
        if approximation_degree < 0.0 or approximation_degree > 1.0:
            raise TranspilerError("Approximation degree must be in [0.0, 1.0]")
    else:
        if not all(0.0 <= d <= 1.0 for d in approximation_degree if d):
            raise TranspilerError("Approximation degree must be in [0.0, 1.0]")
    return approximation_degree


def _parse_target(backend, target):
    backend_target = getattr(backend, "target", None)
    if target is None:
        target = backend_target
    return target


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
    if isinstance(timing_constraints, TimingConstraints):
        return [timing_constraints] * num_circuits
    if backend is None and timing_constraints is None:
        timing_constraints = TimingConstraints()
    else:
        backend_version = getattr(backend, "version", 0)
        if backend_version <= 1:
            if timing_constraints is None:
                # get constraints from backend
                timing_constraints = getattr(backend.configuration(), "timing_constraints", {})
            timing_constraints = TimingConstraints(**timing_constraints)
        else:
            timing_constraints = backend.target.timing_constraints()
    return [timing_constraints] * num_circuits


def _zip_dict(mapping: Dict[Any, Iterable]) -> Iterable[Dict]:
    """Zip a dictionary where all the values are iterables of the same length into an iterable of
    dictionaries with the same keys.  This has the same semantics as zip with regard to laziness
    (over the iterables; there must be a finite number of keys!) and unequal lengths."""
    keys, iterables = zip(*mapping.items())
    return (dict(zip(keys, values)) for values in zip(*iterables))
