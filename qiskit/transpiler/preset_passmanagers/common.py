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

# pylint: disable=invalid-name

"""Common preset passmanager generators."""

from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel

from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.passes import Unroller
from qiskit.transpiler.passes import BasisTranslator
from qiskit.transpiler.passes import UnrollCustomDefinitions
from qiskit.transpiler.passes import Unroll3qOrMore
from qiskit.transpiler.passes import Collect2qBlocks
from qiskit.transpiler.passes import ConsolidateBlocks
from qiskit.transpiler.passes import UnitarySynthesis
from qiskit.transpiler.passes import CheckMap
from qiskit.transpiler.passes import GateDirection
from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements
from qiskit.transpiler.passes import CheckGateDirection
from qiskit.transpiler.passes import TimeUnitConversion
from qiskit.transpiler.passes import ALAPSchedule
from qiskit.transpiler.passes import ASAPSchedule
from qiskit.transpiler.passes import FullAncillaAllocation
from qiskit.transpiler.passes import EnlargeWithAncilla
from qiskit.transpiler.passes import ApplyLayout
from qiskit.transpiler.passes import RemoveResetInZeroState
from qiskit.transpiler.passes import ValidatePulseGates
from qiskit.transpiler.passes import AlignMeasures
from qiskit.transpiler.passes import PulseGates
from qiskit.transpiler.exceptions import TranspilerError


def generate_embed_passmanager(coupling_map):
    """Generate a layout embedding :class:`~qiskit.transpiler.PassManager`

    This is used to generate a :class:`~qiskit.transpiler.PassManager` object
    that can be used to expand and apply an initial layout to a circuit

    Args:
        coupling_map (CouplingMap): The coupling map for the backend to embed
            the circuit to.
    Returns:
        PassManager: The embedding passmanager that assumes the layout property
            set has been set in earlier stages
    """
    return PassManager([FullAncillaAllocation(coupling_map), EnlargeWithAncilla(), ApplyLayout()])


def generate_routing_passmanager(routing_pass, coupling_map):
    """Generate a routing :class:`~qiskit.transpiler.PassManager`

    Args:
        routing_pass (TransformationPass): The pass which will perform the
            routing
        coupling_map (CouplingMap): The coupling map of the backend to route
            for
    Returns:
        PassManager: The routing pass manager
    """
    routing = PassManager()
    routing.append(Unroll3qOrMore())
    routing.append(CheckMap(coupling_map))

    def _swap_condition(property_set):
        return not property_set["is_swap_mapped"]

    routing.append([BarrierBeforeFinalMeasurements(), routing_pass], condition=_swap_condition)
    return routing


def generate_pre_op_passmanager(coupling_map=None, remove_reset_in_zero=False):
    """Generate a pre-optimization loop :class:`~qiskit.transpiler.PassManager`

    This pass manager will check to ensure that directionality from the coupling
    map is respected

    Args:
        coupling_map (CouplingMap): The coupling map to use
        remove_reset_in_zero (bool): If ``True`` include the remove reset in
            zero pass in the generated PassManager
    Returns:
        PassManager: The pass manager

    """
    pre_opt = PassManager()
    if coupling_map:
        pre_opt.append(CheckGateDirection(coupling_map))

        def _direction_condition(property_set):
            return not property_set["is_direction_mapped"]

        pre_opt.append([GateDirection(coupling_map)], condition=_direction_condition)
    if remove_reset_in_zero:
        pre_opt.append(RemoveResetInZeroState())
    return pre_opt


def generate_translation_passmanager(
    basis_gates, method="basis", approximation_degree=None, coupling_map=None, backend_props=None
):
    """Generate a basis translation :class:`~qiskit.transpiler.PassManager`

    Args:
        basis_gates (list): A list
        method (str): The basis translation method to use
        approximation_degree (float): The heuristic approximation degree to
            use. Can be between 0 and 1.
        coupling_map (CouplingMap): the coupling map of the backend
            in case synthesis is done on a physical circuit. The
            directionality of the coupling_map will be taken into
            account if pulse_optimize is True/None and natural_direction
            is True/None.
        backend_props (BackendProperties): Properties of a backend to
            synthesize for (e.g. gate fidelities).

    Returns:
        PassManager: The basis translation pass manager

    Raises:
        TranspilerError: If the ``method`` kwarg is not a valid value
    """
    if method == "unroller":
        unroll = [Unroller(basis_gates)]
    elif method == "translator":
        unroll = [UnrollCustomDefinitions(sel, basis_gates), BasisTranslator(sel, basis_gates)]
    elif method == "synthesis":
        unroll = [
            Unroll3qOrMore(),
            Collect2qBlocks(),
            ConsolidateBlocks(basis_gates=basis_gates),
            UnitarySynthesis(
                basis_gates,
                approximation_degree=approximation_degree,
                coupling_map=coupling_map,
                backend_props=backend_props,
            ),
        ]
    else:
        raise TranspilerError("Invalid translation method %s." % method)
    return PassManager(unroll)


def generate_scheduling_post_opt(
    instruction_durations, scheduling_method, timing_constraints, inst_map
):
    """Generate a post optimization scheduling :class:`~qiskit.transpiler.PassManager`

    Args:
        instruction_durations (dict): The dictionary of instruction durations
        scheduling_method (str): The scheduling method to use, can either be
            ``'asap'``/``'as_soon_as_possible'`` or
            ``'alap'``/``'as_late_as_possible'``
        timing_constraints (TimingConstraints): Hardware time alignment restrictions.
        inst_map (InstructionScheduleMap): Mapping object that maps gate to schedule.

    Returns:
        PassManager: The scheduling pass manager

    Raises:
        TranspilerError: If the ``scheduling_method`` kwarg is not a valid value
    """
    scheduling = PassManager()
    if inst_map and inst_map.has_custom_gate():
        scheduling.append(PulseGates(inst_map=inst_map))
    scheduling.append(TimeUnitConversion(instruction_durations))
    if scheduling_method:
        if scheduling_method in {"alap", "as_late_as_possible"}:
            scheduling.append(ALAPSchedule(instruction_durations))
        elif scheduling_method in {"asap", "as_soon_as_possible"}:
            scheduling.append(ASAPSchedule(instruction_durations))
        else:
            raise TranspilerError("Invalid scheduling method %s." % scheduling_method)
    scheduling.append(
        ValidatePulseGates(
            granularity=timing_constraints.granularity, min_length=timing_constraints.min_length
        )
    )
    scheduling.append(AlignMeasures(alignment=timing_constraints.acquire_alignment))
    return scheduling
