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

"""
==================================================================
Preset Passmanagers (:mod:`qiskit.transpiler.preset_passmanagers`)
==================================================================

.. currentmodule:: qiskit.transpiler.preset_passmanagers

This module contains functions for generating the preset pass managers
for the transpiler. The preset pass managers are instances of
:class:`~.StagedPassManager` which are used to execute the circuit
transformations as part of Qiskit's compiler inside the
:func:`~.transpile` function at the different optimization levels.
The functionality here is divided into two parts, the first includes the
functions used generate the entire pass manager which is used by
:func:`~.transpile` (:ref:`preset_pass_manager_generators`) and the
second includes functions which are used to build (either entirely or in
part) the stages which the preset pass managers are composed of
(:ref:`stage_generators`).

.. _preset_pass_manager_generators:

Preset Pass Manager Generation
------------------------------

.. autofunction:: generate_preset_pass_manager
.. autofunction:: level_0_pass_manager
.. autofunction:: level_1_pass_manager
.. autofunction:: level_2_pass_manager
.. autofunction:: level_3_pass_manager

.. _stage_generators:

Stage Generator Functions
-------------------------

.. currentmodule:: qiskit.transpiler.preset_passmanagers.common
.. autofunction:: generate_control_flow_options_check
.. autofunction:: generate_error_on_control_flow
.. autofunction:: generate_unroll_3q
.. autofunction:: generate_embed_passmanager
.. autofunction:: generate_routing_passmanager
.. autofunction:: generate_pre_op_passmanager
.. autofunction:: generate_translation_passmanager
.. autofunction:: generate_scheduling
.. currentmodule:: qiskit.transpiler.preset_passmanagers
"""

from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.target import target_to_backend_properties

from .level0 import level_0_pass_manager
from .level1 import level_1_pass_manager
from .level2 import level_2_pass_manager
from .level3 import level_3_pass_manager


def generate_preset_pass_manager(
    optimization_level,
    backend=None,
    target=None,
    basis_gates=None,
    inst_map=None,
    coupling_map=None,
    instruction_durations=None,
    backend_properties=None,
    timing_constraints=None,
    initial_layout=None,
    layout_method=None,
    routing_method=None,
    translation_method=None,
    scheduling_method=None,
    approximation_degree=None,
    seed_transpiler=None,
    unitary_synthesis_method="default",
    unitary_synthesis_plugin_config=None,
    hls_config=None,
    init_method=None,
    optimization_method=None,
    *,
    _skip_target=False,
):
    """Generate a preset :class:`~.PassManager`

    This function is used to quickly generate a preset pass manager. A preset pass
    manager are the default pass managers used by the :func:`~.transpile`
    function. This function provides a convenient and simple method to construct
    a standalone :class:`~.PassManager` object that mirrors what the transpile


    Args:
        optimization_level (int): The optimization level to generate a
            :class:`~.PassManager` for. This can be 0, 1, 2, or 3. Higher
            levels generate more optimized circuits, at the expense of
            longer transpilation time:

                * 0: no optimization
                * 1: light optimization
                * 2: heavy optimization
                * 3: even heavier optimization

        backend (Backend): An optional backend object which can be used as the
            source of the default values for the ``basis_gates``, ``inst_map``,
            ``couplig_map``, ``backend_properties``, ``instruction_durations``,
            ``timing_constraints``, and ``target``. If any of those other arguments
            are specified in addition to ``backend`` they will take precedence
            over the value contained in the backend.
        target (Target): The :class:`~.Target` representing a backend compilation
            target. The following attributes will be inferred from this
            argument if they are not set: ``coupling_map``, ``basis_gates``,
            ``instruction_durations``, ``inst_map``, ``timing_constraints``
            and ``backend_properties``.
        basis_gates (list): List of basis gate names to unroll to
            (e.g: ``['u1', 'u2', 'u3', 'cx']``).
        inst_map (InstructionScheduleMap): Mapping object that maps gate to schedules.
            If any user defined calibration is found in the map and this is used in a
            circuit, transpiler attaches the custom gate definition to the circuit.
            This enables one to flexibly override the low-level instruction
            implementation.
        coupling_map (CouplingMap): Directed graph represented a coupling
            map.
        instruction_durations (InstructionDurations): Dictionary of duration
            (in dt) for each instruction.
        timing_constraints (TimingConstraints): Hardware time alignment restrictions.
        initial_layout (Layout): Initial position of virtual qubits on
            physical qubits.
        layout_method (str): The :class:`~.Pass` to use for choosing initial qubit
            placement. Valid choices are ``'trivial'``, ``'dense'``, ``'noise_adaptive'``,
            and, ``'sabre'`` representing :class:`~.TrivialLayout`, :class:`~DenseLayout`,
            :class:`~.NoiseAdaptiveLayout`, :class:`~.SabreLayout` respectively. This can also
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
            basis gates. Valid choices ``'unroller'``, ``'translator'``, ``'synthesis'``
            representing :class:`~.Unroller`, :class:`~.BasisTranslator`, and
            :class:`~.UnitarySynthesis` respectively. This can also be the external plugin
            name to use for the ``translation`` stage of the output :class:`~.StagedPassManager`.
            You can see a list of installed plugins by using :func:`~.list_stage_plugins` with
            ``"translation"`` for the ``stage_name`` argument.
        scheduling_method (str): The pass to use for scheduling instructions. Valid choices
            are ``'alap'`` and ``'asap'``. This can also be the external plugin name to use
            for the ``scheduling`` stage of the output :class:`~.StagedPassManager`. You can
            see a list of installed plugins by using :func:`~.list_stage_plugins` with
            ``"scheduling"`` for the ``stage_name`` argument.
        backend_properties (BackendProperties): Properties returned by a
            backend, including information on gate errors, readout errors,
            qubit coherence times, etc.
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
            the ``unitary_synthesis`` argument. As this is custom for each
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

    Returns:
        StagedPassManager: The preset pass manager for the given options

    Raises:
        ValueError: if an invalid value for ``optimization_level`` is passed in.
    """
    if target is not None:
        if coupling_map is None:
            coupling_map = target.build_coupling_map()
        if basis_gates is None:
            basis_gates = target.operation_names
        if instruction_durations is None:
            instruction_durations = target.durations()
        if inst_map is None:
            inst_map = target.instruction_schedule_map()
        if timing_constraints is None:
            timing_constraints = target.timing_constraints()
        if backend_properties is None:
            backend_properties = target_to_backend_properties(target)

    pm_options = {
        "target": target,
        "basis_gates": basis_gates,
        "inst_map": inst_map,
        "coupling_map": coupling_map,
        "instruction_durations": instruction_durations,
        "backend_properties": backend_properties,
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


__all__ = [
    "level_0_pass_manager",
    "level_1_pass_manager",
    "level_2_pass_manager",
    "level_3_pass_manager",
    "generate_preset_pass_manager",
]
