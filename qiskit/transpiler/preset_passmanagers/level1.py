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

"""Pass manager for optimization level 1, providing light optimization.

Level 1 pass manager: light optimization by simple adjacent gate collapsing.
"""

from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.timing_constraints import TimingConstraints
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.passmanager import FullPassManager

from qiskit.transpiler.passes import CXCancellation
from qiskit.transpiler.passes import SetLayout
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
from qiskit.transpiler.passes import Layout2qDistance
from qiskit.transpiler.passes import Error
from qiskit.transpiler.preset_passmanagers import common

from qiskit.transpiler import TranspilerError


def level_1_pass_manager(pass_manager_config: PassManagerConfig) -> PassManager:
    """Level 1 pass manager: light optimization by simple adjacent gate collapsing.

    This pass manager applies the user-given initial layout. If none is given,
    and a trivial layout (i-th virtual -> i-th physical) makes the circuit fit
    the coupling map, that is used.
    Otherwise, the circuit is mapped to the most densely connected coupling subgraph,
    and swaps are inserted to map. Any unused physical qubit is allocated as ancilla space.
    The pass manager then unrolls the circuit to the desired basis, and transforms the
    circuit to match the coupling map. Finally, optimizations in the form of adjacent
    gate collapse and redundant reset removal are performed.

    Note:
        In simulators where ``coupling_map=None``, only the unrolling and
        optimization stages are done.

    Args:
        pass_manager_config: configuration of the pass manager.

    Returns:
        a level 1 pass manager.

    Raises:
        TranspilerError: if the passmanager config is invalid.
    """
    basis_gates = pass_manager_config.basis_gates
    inst_map = pass_manager_config.inst_map
    coupling_map = pass_manager_config.coupling_map
    initial_layout = pass_manager_config.initial_layout
    layout_method = pass_manager_config.layout_method or "dense"
    routing_method = pass_manager_config.routing_method or "stochastic"
    translation_method = pass_manager_config.translation_method or "translator"
    scheduling_method = pass_manager_config.scheduling_method
    instruction_durations = pass_manager_config.instruction_durations
    seed_transpiler = pass_manager_config.seed_transpiler
    backend_properties = pass_manager_config.backend_properties
    approximation_degree = pass_manager_config.approximation_degree
    timing_constraints = pass_manager_config.timing_constraints or TimingConstraints()

    # Use trivial layout if no layout given
    _given_layout = SetLayout(initial_layout)

    _choose_layout_and_score = [
        TrivialLayout(coupling_map),
        Layout2qDistance(coupling_map, property_name="trivial_layout_score"),
    ]

    def _choose_layout_condition(property_set):
        return not property_set["layout"]

    # Use a better layout on densely connected qubits, if circuit needs swaps
    if layout_method == "trivial":
        _improve_layout = TrivialLayout(coupling_map)
    elif layout_method == "dense":
        _improve_layout = DenseLayout(coupling_map, backend_properties)
    elif layout_method == "noise_adaptive":
        _improve_layout = NoiseAdaptiveLayout(backend_properties)
    elif layout_method == "sabre":
        _improve_layout = SabreLayout(coupling_map, max_iterations=2, seed=seed_transpiler)
    else:
        raise TranspilerError("Invalid layout method %s." % layout_method)

    def _not_perfect_yet(property_set):
        return (
            property_set["trivial_layout_score"] is not None
            and property_set["trivial_layout_score"] != 0
        )

    if routing_method == "basic":
        routing_pass = BasicSwap(coupling_map)
    elif routing_method == "stochastic":
        routing_pass = StochasticSwap(coupling_map, trials=20, seed=seed_transpiler)
    elif routing_method == "lookahead":
        routing_pass = LookaheadSwap(coupling_map, search_depth=4, search_width=4)
    elif routing_method == "sabre":
        routing_pass = SabreSwap(coupling_map, heuristic="lookahead", seed=seed_transpiler)
    elif routing_method == "none":
        routing_pass = Error(
            msg="No routing method selected, but circuit is not routed to device. "
            "CheckMap Error: {check_map_msg}",
            action="raise",
        )
    else:
        raise TranspilerError("Invalid routing method %s." % routing_method)

    # Build optimization loop: merge 1q rotations and cancel CNOT gates iteratively
    # until no more change in depth
    _depth_check = [Depth(), FixedPoint("depth")]

    def _opt_control(property_set):
        return not property_set["depth_fixed_point"]

    _opt = [Optimize1qGatesDecomposition(basis_gates), CXCancellation()]

    # Build full pass manager
    if coupling_map or initial_layout:
        layout = PassManager()
        layout.append(_given_layout)
        layout.append(_choose_layout_and_score, condition=_choose_layout_condition)
        layout.append(_improve_layout, condition=_not_perfect_yet)
        layout += common.generate_embed_passmanager(coupling_map)
        routing = common.generate_routing_passmanager(routing_pass, coupling_map)
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

    return FullPassManager(
        layout=layout,
        routing=routing,
        translation=translation,
        pre_optimization=pre_optimization,
        optimization=optimization,
        scheduling=sched,
    )
