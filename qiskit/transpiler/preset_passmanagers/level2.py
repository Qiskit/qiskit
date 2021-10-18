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

"""Pass manager for optimization level 2, providing medium optimization.

Level 2 pass manager: medium optimization by noise adaptive qubit mapping and
gate cancellation using commutativity rules.
"""

from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.timing_constraints import TimingConstraints
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.passmanager import StructuredPassManager

from qiskit.transpiler.passes import SetLayout
from qiskit.transpiler.passes import CSPLayout
from qiskit.transpiler.passes import TrivialLayout
from qiskit.transpiler.passes import DenseLayout
from qiskit.transpiler.passes import NoiseAdaptiveLayout
from qiskit.transpiler.passes import SabreLayout
from qiskit.transpiler.passes import BasicSwap
from qiskit.transpiler.passes import LookaheadSwap
from qiskit.transpiler.passes import StochasticSwap
from qiskit.transpiler.passes import SabreSwap
from qiskit.transpiler.passes import FixedPoint
from qiskit.transpiler.passes import Depth
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
from qiskit.transpiler.passes import CommutativeCancellation
from qiskit.transpiler.passes import Layout2qDistance
from qiskit.transpiler.passes import Error
from qiskit.transpiler.preset_passmanagers import common

from qiskit.transpiler import TranspilerError


def level_2_pass_manager(pass_manager_config: PassManagerConfig) -> PassManager:
    """Level 2 pass manager: medium optimization by initial layout selection and
    gate cancellation using commutativity rules.

    This pass manager applies the user-given initial layout. If none is given, a search
    for a perfect layout (i.e. one that satisfies all 2-qubit interactions) is conducted.
    If no such layout is found, qubits are laid out on the most densely connected subset
    which also exhibits the best gate fidelities.

    The pass manager then transforms the circuit to match the coupling constraints.
    It is then unrolled to the basis, and any flipped cx directions are fixed.
    Finally, optimizations in the form of commutative gate cancellation and redundant
    reset removal are performed.

    Note:
        In simulators where ``coupling_map=None``, only the unrolling and
        optimization stages are done.

    Args:
        pass_manager_config: configuration of the pass manager.

    Returns:
        a level 2 pass manager.

    Raises:
        TranspilerError: if the passmanager config is invalid.
    """
    basis_gates = pass_manager_config.basis_gates
    inst_map = pass_manager_config.inst_map
    coupling_map = pass_manager_config.coupling_map
    initial_layout = pass_manager_config.initial_layout
    layout_method = pass_manager_config.layout_method or "dense"
    routing_method = pass_manager_config.routing_method or "stochastic"
    translation_method = pass_manager_config.translation_method or "basis_translator"
    scheduling_method = pass_manager_config.scheduling_method
    instruction_durations = pass_manager_config.instruction_durations
    seed_transpiler = pass_manager_config.seed_transpiler
    backend_properties = pass_manager_config.backend_properties
    approximation_degree = pass_manager_config.approximation_degree
    unitary_synthesis_method = pass_manager_config.unitary_synthesis_method
    timing_constraints = pass_manager_config.timing_constraints or TimingConstraints()

    # Search for a perfect layout, or choose a dense layout, if no layout given
    _given_layout = SetLayout(initial_layout)

    def _choose_layout_condition(property_set):
        # layout hasn't been set yet
        return not property_set["layout"]

    # If layout_method is not set, first try a trivial layout
    _choose_layout_0 = (
        []
        if pass_manager_config.layout_method
        else [
            TrivialLayout(coupling_map),
            Layout2qDistance(coupling_map, property_name="trivial_layout_score"),
        ]
    )
    # If a trivial layout wasn't perfect (ie no swaps are needed) then try using
    # CSP layout to find a perfect layout
    _choose_layout_1 = (
        []
        if pass_manager_config.layout_method
        else CSPLayout(coupling_map, call_limit=1000, time_limit=10, seed=seed_transpiler)
    )

    def _trivial_not_perfect(property_set):
        # Verify that a trivial layout  is perfect. If trivial_layout_score > 0
        # the layout is not perfect. The layout is unconditionally set by trivial
        # layout so we need to clear it before contuing.
        if property_set["trivial_layout_score"] is not None:
            if property_set["trivial_layout_score"] != 0:
                property_set["layout"]._wrapped = None
                return True
        return False

    def _csp_not_found_match(property_set):
        # If a layout hasn't been set by the time we run csp we need to run layout
        if property_set["layout"] is None:
            return True
        # if CSP layout stopped for any reason other than solution found we need
        # to run layout since CSP didn't converge.
        if (
            property_set["CSPLayout_stop_reason"] is not None
            and property_set["CSPLayout_stop_reason"] != "solution found"
        ):
            return True
        return False

    # if CSP layout doesn't converge on a solution use layout_method (dense) to get a layout
    if layout_method == "trivial":
        _choose_layout_2 = TrivialLayout(coupling_map)
    elif layout_method == "dense":
        _choose_layout_2 = DenseLayout(coupling_map, backend_properties)
    elif layout_method == "noise_adaptive":
        _choose_layout_2 = NoiseAdaptiveLayout(backend_properties)
    elif layout_method == "sabre":
        _choose_layout_2 = SabreLayout(coupling_map, max_iterations=2, seed=seed_transpiler)
    else:
        raise TranspilerError("Invalid layout method %s." % layout_method)

    if routing_method == "basic":
        routing_pass = BasicSwap(coupling_map)
    elif routing_method == "stochastic":
        routing_pass = StochasticSwap(coupling_map, trials=20, seed=seed_transpiler)
    elif routing_method == "lookahead":
        routing_pass = LookaheadSwap(coupling_map, search_depth=5, search_width=5)
    elif routing_method == "sabre":
        routing_pass = SabreSwap(coupling_map, heuristic="decay", seed=seed_transpiler)
    elif routing_method == "none":
        routing_pass = Error(
            msg="No routing method selected, but circuit is not routed to device. "
            "CheckMap Error: {check_map_msg}",
            action="raise",
        )
    else:
        raise TranspilerError("Invalid routing method %s." % routing_method)

    # Build optimization loop: 1q rotation merge and commutative cancellation iteratively until
    # no more change in depth
    _depth_check = [Depth(), FixedPoint("depth")]

    def _opt_control(property_set):
        return not property_set["depth_fixed_point"]

    _opt = [
        Optimize1qGatesDecomposition(basis_gates),
        CommutativeCancellation(basis_gates=basis_gates),
    ]

    # Build pass manager
    if coupling_map or initial_layout:
        layout = PassManager()
        layout.append(_given_layout)
        layout.append(_choose_layout_0, condition=_choose_layout_condition)
        layout.append(_choose_layout_1, condition=_trivial_not_perfect)
        layout.append(_choose_layout_2, condition=_csp_not_found_match)
        layout += common.generate_embed_passmanager(coupling_map)
        routing = common.generate_routing_passmanager(
            routing_pass,
            coupling_map,
            basis_gates,
            approximation_degree,
            backend_properties,
            unitary_synthesis_method,
        )
    else:
        layout = None
        routing = None
    translation = common.generate_translation_passmanager(
        basis_gates, translation_method, approximation_degree, coupling_map, backend_properties
    )
    if coupling_map and not coupling_map.is_symmetric:
        pre_optimization = common.generate_pre_op_passmanager(coupling_map, True)
    else:
        pre_optimization = common.generate_pre_op_passmanager(remove_reset_in_zero=True)
    optimization = PassManager()
    unroll = [pass_ for x in translation.passes() for pass_ in x["passes"]]
    opt_loop = _depth_check + _opt + unroll
    optimization.append(opt_loop, do_while=_opt_control)
    sched = common.generate_scheduling_post_opt(
        instruction_durations, scheduling_method, timing_constraints, inst_map
    )

    return StructuredPassManager(
        layout=layout,
        routing=routing,
        translation=translation,
        pre_optimization=pre_optimization,
        optimization=optimization,
        scheduling=sched,
    )
