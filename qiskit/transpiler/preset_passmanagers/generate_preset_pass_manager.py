# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Preset pass manager generation function
"""
import copy
import warnings

from qiskit.circuit.controlflow import CONTROL_FLOW_OP_NAMES, get_control_flow_name_mapping
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.circuit import Qubit
from qiskit.providers.backend import Backend
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.target import Target, _FakeTarget
from qiskit.transpiler.timing_constraints import TimingConstraints

from .level0 import level_0_pass_manager
from .level1 import level_1_pass_manager
from .level2 import level_2_pass_manager
from .level3 import level_3_pass_manager


OVER_3Q_GATES = ["ccx", "ccz", "cswap", "rccx", "c3x", "c3sx", "rc3x"]


def generate_preset_pass_manager(
    optimization_level=2,
    backend=None,
    target=None,
    basis_gates=None,
    coupling_map=None,
    initial_layout=None,
    layout_method=None,
    routing_method=None,
    translation_method=None,
    scheduling_method=None,
    approximation_degree=1.0,
    seed_transpiler=None,
    unitary_synthesis_method="default",
    unitary_synthesis_plugin_config=None,
    hls_config=None,
    init_method=None,
    optimization_method=None,
    dt=None,
    qubits_initially_zero=True,
    *,
    _skip_target=False,
):
    """Generate a preset :class:`~.PassManager`

    This function is used to quickly generate a preset pass manager. Preset pass
    managers are the default pass managers used by the :func:`~.transpile`
    function. This function provides a convenient and simple method to construct
    a standalone :class:`~.PassManager` object that mirrors what the :func:`~.transpile`
    function internally builds and uses.

    The target constraints for the pass manager construction can be specified through a :class:`.Target`
    instance, a :class:`.BackendV2` instance, or via loose constraints
    (``basis_gates``, ``coupling_map``, or ``dt``).
    The order of priorities for target constraints works as follows: if a ``target``
    input is provided, it will take priority over any ``backend`` input or loose constraints.
    If a ``backend`` is provided together with any loose constraint
    from the list above, the loose constraint will take priority over the corresponding backend
    constraint. This behavior is summarized in the table below. The first column
    in the table summarizes the potential user-provided constraints, and each cell shows whether
    the priority is assigned to that specific constraint input or another input
    (`target`/`backend(V1)`/`backend(V2)`).

    ============================ ========= ========================
    User Provided                target    backend(V2)
    ============================ ========= ========================
    **basis_gates**              target    basis_gates
    **coupling_map**             target    coupling_map
    **dt**                       target    dt
    ============================ ========= ========================

    Args:
        optimization_level (int): The optimization level to generate a
            :class:`~.StagedPassManager` for. By default optimization level 2
            is used if this is not specified. This can be 0, 1, 2, or 3. Higher
            levels generate potentially more optimized circuits, at the expense
            of longer transpilation time:

                * 0: no optimization
                * 1: light optimization
                * 2: heavy optimization
                * 3: even heavier optimization

        backend (Backend): An optional backend object which can be used as the
            source of the default values for the ``basis_gates``,
            ``coupling_map``, and ``target``. If any of those other arguments
            are specified in addition to ``backend`` they will take precedence
            over the value contained in the backend.
        target (Target): The :class:`~.Target` representing a backend compilation
            target. The following attributes will be inferred from this
            argument if they are not set: ``coupling_map`` and ``basis_gates``.
        basis_gates (list): List of basis gate names to unroll to
            (e.g: ``['u1', 'u2', 'u3', 'cx']``).
        coupling_map (CouplingMap or list): Directed graph represented a coupling
            map. Multiple formats are supported:

            #. ``CouplingMap`` instance
            #. List, must be given as an adjacency matrix, where each entry
               specifies all directed two-qubit interactions supported by backend,
               e.g: ``[[0, 1], [0, 3], [1, 2], [1, 5], [2, 5], [4, 1], [5, 3]]``
        dt (float): Backend sample time (resolution) in seconds.
            If ``None`` (default) and a backend is provided, ``backend.dt`` is used.
        initial_layout (Layout | List[int]): Initial position of virtual qubits on
            physical qubits.
        layout_method (str): The :class:`~.Pass` to use for choosing initial qubit
            placement. Valid choices are ``'trivial'``, ``'dense'``,
            and ``'sabre'``, representing :class:`~.TrivialLayout`, :class:`~.DenseLayout` and
            :class:`~.SabreLayout` respectively. This can also
            be the external plugin name to use for the ``layout`` stage of the output
            :class:`~.StagedPassManager`. You can see a list of installed plugins by using
            :func:`~.list_stage_plugins` with ``"layout"`` for the ``stage_name`` argument.
        routing_method (str): The pass to use for routing qubits on the
            architecture. Valid choices are ``'basic'``, ``'lookahead'``, ``'stochastic'``,
            ``'sabre'``, and ``'none'`` representing :class:`~.BasicSwap`,
            :class:`~.LookaheadSwap`, :class:`~.StochasticSwap`, :class:`~.SabreSwap`, and
            erroring if routing is required respectively. This can also be the external plugin
            name to use for the ``routing`` stage of the output :class:`~.StagedPassManager`.
            You can see a list of installed plugins by using :func:`~.list_stage_plugins` with
            ``"routing"`` for the ``stage_name`` argument.
        translation_method (str): The method to use for translating gates to
            basis gates. Valid choices ``'translator'``, ``'synthesis'`` representing
            :class:`~.BasisTranslator`, and :class:`~.UnitarySynthesis` respectively. This can
            also be the external plugin name to use for the ``translation`` stage of the output
            :class:`~.StagedPassManager`. You can see a list of installed plugins by using
            :func:`~.list_stage_plugins` with ``"translation"`` for the ``stage_name`` argument.
        scheduling_method (str): The pass to use for scheduling instructions. Valid choices
            are ``'alap'`` and ``'asap'``. This can also be the external plugin name to use
            for the ``scheduling`` stage of the output :class:`~.StagedPassManager`. You can
            see a list of installed plugins by using :func:`~.list_stage_plugins` with
            ``"scheduling"`` for the ``stage_name`` argument.
        approximation_degree (float): Heuristic dial used for circuit approximation
            (1.0=no approximation, 0.0=maximal approximation).
        seed_transpiler (int): Sets random seed for the stochastic parts of
            the transpiler.
        unitary_synthesis_method (str): The name of the unitary synthesis
            method to use. By default ``'default'`` is used. You can see a list of
            installed plugins with :func:`.unitary_synthesis_plugin_names`.
        unitary_synthesis_plugin_config (dict): An optional configuration dictionary
            that will be passed directly to the unitary synthesis plugin. By
            default this setting will have no effect as the default unitary
            synthesis method does not take custom configuration. This should
            only be necessary when a unitary synthesis plugin is specified with
            the ``unitary_synthesis_method`` argument. As this is custom for each
            unitary synthesis plugin refer to the plugin documentation for how
            to use this option.
        hls_config (HLSConfig): An optional configuration class :class:`~.HLSConfig`
            that will be passed directly to :class:`~.HighLevelSynthesis` transformation pass.
            This configuration class allows to specify for various high-level objects
            the lists of synthesis algorithms and their parameters.
        init_method (str): The plugin name to use for the ``init`` stage of
            the output :class:`~.StagedPassManager`. By default an external
            plugin is not used. You can see a list of installed plugins by
            using :func:`~.list_stage_plugins` with ``"init"`` for the stage
            name argument.
        optimization_method (str): The plugin name to use for the
            ``optimization`` stage of the output
            :class:`~.StagedPassManager`. By default an external
            plugin is not used. You can see a list of installed plugins by
            using :func:`~.list_stage_plugins` with ``"optimization"`` for the
            ``stage_name`` argument.
        qubits_initially_zero (bool): Indicates whether the input circuit is
                zero-initialized.

    Returns:
        StagedPassManager: The preset pass manager for the given options

    Raises:
        ValueError: if an invalid value for ``optimization_level`` is passed in.
    """

    # Handle positional arguments for target and backend. This enables the usage
    # pattern `generate_preset_pass_manager(backend.target)` to generate a default
    # pass manager for a given target.
    if isinstance(optimization_level, Target):
        target = optimization_level
        optimization_level = 2
    elif isinstance(optimization_level, Backend):
        backend = optimization_level
        optimization_level = 2

    # If there are no loose constraints => use backend target if available
    _no_loose_constraints = basis_gates is None and coupling_map is None and dt is None

    # If the only loose constraint is dt => use backend target and modify dt
    _adjust_dt = backend is not None and dt is not None

    # Warn about inconsistencies in backend + loose constraints path (dt shouldn't be a problem)
    if backend is not None and (coupling_map is not None or basis_gates is not None):
        warnings.warn(
            "Providing `coupling_map` and/or `basis_gates` along with `backend` is not "
            "recommended, as this will invalidate the backend's gate durations and error rates.",
            category=UserWarning,
            stacklevel=2,
        )

    # Resolve loose constraints case-by-case against backend constraints.
    # The order of priority is loose constraints > backend.
    dt = _parse_dt(dt, backend)
    instruction_durations = _parse_instruction_durations(backend, dt)
    timing_constraints = _parse_timing_constraints(backend)
    coupling_map = _parse_coupling_map(coupling_map, backend)
    basis_gates, name_mapping = _parse_basis_gates(basis_gates, backend)

    # Check if coupling map has been provided (either standalone or through backend)
    # with user-defined basis_gates, and whether these have 3q or more.
    if coupling_map is not None and basis_gates is not None:
        for gate in OVER_3Q_GATES:
            if gate in basis_gates:
                raise ValueError(
                    f"Gates with 3 or more qubits ({gate}) in `basis_gates` or `backend` are "
                    "incompatible with a custom `coupling_map`. To include 3-qubit or larger "
                    " gates in the transpilation basis, provide a custom `target` instead."
                )

    if target is None:
        if backend is not None and _no_loose_constraints:
            # If a backend is specified without loose constraints, use its target directly.
            target = backend.target
        elif _adjust_dt:
            # If a backend is specified with loose dt, use its target and adjust the dt value.
            target = copy.deepcopy(backend.target)
            target.dt = dt
        else:
            if basis_gates is not None:
                # Build target from constraints.
                target = Target.from_configuration(
                    basis_gates=basis_gates,
                    num_qubits=backend.num_qubits if backend is not None else None,
                    coupling_map=coupling_map,
                    instruction_durations=instruction_durations,
                    concurrent_measurements=(
                        backend.target.concurrent_measurements if backend is not None else None
                    ),
                    dt=dt,
                    timing_constraints=timing_constraints,
                    custom_name_mapping=name_mapping,
                )
            else:
                target = _FakeTarget.from_configuration(
                    num_qubits=backend.num_qubits if backend is not None else None,
                    coupling_map=coupling_map,
                    dt=dt,
                )

    # Update loose constraints to populate pm options
    if coupling_map is None:
        coupling_map = target.build_coupling_map()
    if basis_gates is None and len(target.operation_names) > 0:
        basis_gates = target.operation_names
    if instruction_durations is None:
        instruction_durations = target.durations()
    if timing_constraints is None:
        timing_constraints = target.timing_constraints()

    # Parse non-target dependent pm options
    initial_layout = _parse_initial_layout(initial_layout)
    approximation_degree = _parse_approximation_degree(approximation_degree)
    seed_transpiler = _parse_seed_transpiler(seed_transpiler)

    pm_options = {
        "target": target,
        "basis_gates": basis_gates,
        "coupling_map": coupling_map,
        "instruction_durations": instruction_durations,
        "timing_constraints": timing_constraints,
        "layout_method": layout_method,
        "routing_method": routing_method,
        "translation_method": translation_method,
        "scheduling_method": scheduling_method,
        "approximation_degree": approximation_degree,
        "seed_transpiler": seed_transpiler,
        "unitary_synthesis_method": unitary_synthesis_method,
        "unitary_synthesis_plugin_config": unitary_synthesis_plugin_config,
        "initial_layout": initial_layout,
        "hls_config": hls_config,
        "init_method": init_method,
        "optimization_method": optimization_method,
        "qubits_initially_zero": qubits_initially_zero,
    }

    if backend is not None:
        pm_options["_skip_target"] = _skip_target
        pm_config = PassManagerConfig.from_backend(backend, **pm_options)
    else:
        pm_config = PassManagerConfig(**pm_options)
    if optimization_level == 0:
        pm = level_0_pass_manager(pm_config)
    elif optimization_level == 1:
        pm = level_1_pass_manager(pm_config)
    elif optimization_level == 2:
        pm = level_2_pass_manager(pm_config)
    elif optimization_level == 3:
        pm = level_3_pass_manager(pm_config)
    else:
        raise ValueError(f"Invalid optimization level {optimization_level}")
    return pm


def _parse_basis_gates(basis_gates, backend):
    standard_gates = get_standard_gate_name_mapping()
    # Add control flow gates by default to basis set and name mapping
    default_gates = {"measure", "delay", "reset"}.union(CONTROL_FLOW_OP_NAMES)
    name_mapping = get_control_flow_name_mapping()
    try:
        instructions = set(basis_gates)
        for name in default_gates:
            if name not in instructions:
                instructions.add(name)
    except TypeError:
        instructions = None

    if backend is None:
        # Check for custom instructions
        if instructions is None:
            return None, name_mapping

        for inst in instructions:
            if inst not in standard_gates and inst not in default_gates:
                raise ValueError(
                    f"Providing non-standard gates ({inst}) through the ``basis_gates`` "
                    "argument is not allowed. Use the ``target`` parameter instead. "
                    "You can build a target instance using ``Target.from_configuration()`` and provide "
                    "custom gate definitions with the ``custom_name_mapping`` argument."
                )

        return list(instructions), name_mapping

    instructions = instructions or backend.operation_names
    name_mapping.update(
        {name: backend.target.operation_from_name(name) for name in backend.operation_names}
    )

    # Check for custom instructions
    for inst in instructions:
        if inst not in standard_gates and inst not in default_gates:
            if inst not in backend.operation_names:
                # do not raise error when the custom instruction comes from the backend
                # (common case with BasicSimulator)
                raise ValueError(
                    f"Providing non-standard gates ({inst}) through the ``basis_gates`` "
                    "argument is not allowed. Use the ``target`` parameter instead. "
                    "You can build a target instance using ``Target.from_configuration()`` and provide "
                    "custom gate definitions with the ``custom_name_mapping`` argument."
                )

    return list(instructions) if instructions else None, name_mapping


def _parse_dt(dt, backend):
    # try getting dt from user, else backend
    if dt is None and backend is not None:
        dt = backend.target.dt
    return dt


def _parse_coupling_map(coupling_map, backend):
    # try getting coupling_map from user, else backend
    if coupling_map is None and backend is not None:
        coupling_map = backend.coupling_map

    # coupling_map could be None, or a list of lists, e.g. [[0, 1], [2, 1]]
    if coupling_map is None or isinstance(coupling_map, CouplingMap):
        return coupling_map
    if isinstance(coupling_map, list) and all(
        isinstance(i, list) and len(i) == 2 for i in coupling_map
    ):
        return CouplingMap(coupling_map)
    else:
        raise TranspilerError(
            "Only a single input coupling map can be used with generate_preset_pass_manager()."
        )


def _parse_instruction_durations(backend, dt):
    """Create a list of ``InstructionDuration``s populated from the backend."""
    final_durations = InstructionDurations()
    backend_durations = InstructionDurations()
    if backend is not None:
        backend_durations = backend.instruction_durations
    final_durations.update(backend_durations, dt or backend_durations.dt)
    return final_durations


def _parse_timing_constraints(backend):
    if backend is None:
        timing_constraints = TimingConstraints()
    else:
        timing_constraints = backend.target.timing_constraints()
    return timing_constraints


def _parse_initial_layout(initial_layout):
    # initial_layout could be None, or a list of ints, e.g. [0, 5, 14]
    # or a list of tuples/None e.g. [qr[0], None, qr[1]] or a dict e.g. {qr[0]: 0}
    if initial_layout is None or isinstance(initial_layout, Layout):
        return initial_layout
    if isinstance(initial_layout, dict):
        return Layout(initial_layout)
    initial_layout = list(initial_layout)
    if all(phys is None or isinstance(phys, Qubit) for phys in initial_layout):
        return Layout.from_qubit_list(initial_layout)
    return initial_layout


def _parse_approximation_degree(approximation_degree):
    if approximation_degree is None:
        return None
    if approximation_degree < 0.0 or approximation_degree > 1.0:
        raise TranspilerError("Approximation degree must be in [0.0, 1.0]")
    return approximation_degree


def _parse_seed_transpiler(seed_transpiler):
    if seed_transpiler is None:
        return None
    if not isinstance(seed_transpiler, int) or seed_transpiler < 0:
        raise ValueError("Expected non-negative integer as seed for transpiler.")
    return seed_transpiler
