# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""VF2Layout pass to find a layout using subgraph isomorphism"""

import random
from enum import Enum

from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.layout import vf2_utils
from qiskit._accelerate.vf2_layout import vf2_layout_pass, MultiQEncountered


class VF2LayoutStopReason(Enum):
    """Stop reasons for VF2Layout pass."""

    SOLUTION_FOUND = "solution found"
    NO_SOLUTION_FOUND = "nonexistent solution"
    MORE_THAN_2Q = ">2q gates in basis"


class VF2Layout(AnalysisPass):
    """A pass for choosing a Layout of a circuit onto a Coupling graph, as
    a subgraph isomorphism problem, solved by VF2++.

    If a solution is found that means there is a "perfect layout" and that no
    further swap mapping or routing is needed. If a solution is found the layout
    will be set in the property set as ``property_set['layout']``. However, if no
    solution is found, no ``property_set['layout']`` is set. The stopping reason is
    set in ``property_set['VF2Layout_stop_reason']`` in all the cases and will be
    one of the values enumerated in ``VF2LayoutStopReason`` which has the
    following values:

        * ``"solution found"``: If a perfect layout was found.
        * ``"nonexistent solution"``: If no perfect layout was found.
        * ``">2q gates in basis"``: If VF2Layout can't work with basis

    By default, this pass will construct a heuristic scoring map based on
    the error rates in the provided ``target`` (or ``properties`` if ``target``
    is not provided). However, analysis passes can be run prior to this pass
    and set ``vf2_avg_error_map`` in the property set with a :class:`~.ErrorMap`
    instance. If a value is ``NaN`` that is treated as an ideal edge
    For example if an error map is created as::

        from qiskit.transpiler.passes.layout.vf2_utils import ErrorMap

        error_map = ErrorMap(3)
        error_map.add_error((0, 0), 0.0024)
        error_map.add_error((0, 1), 0.01)
        error_map.add_error((1, 1), 0.0032)

    that represents the error map for a 2 qubit target, where the avg 1q error
    rate is ``0.0024`` on qubit 0 and ``0.0032`` on qubit 1. Then the avg 2q
    error rate for gates that operate on (0, 1) is 0.01 and (1, 0) is not
    supported by the target. This will be used for scoring if it's set as the
    ``vf2_avg_error_map`` key in the property set when :class:`~.VF2Layout` is run.
    """

    def __init__(
        self,
        coupling_map=None,
        strict_direction=False,
        seed=None,
        call_limit=None,
        time_limit=None,
        max_trials=None,
        target=None,
    ):
        """Initialize a ``VF2Layout`` pass instance

        Args:
            coupling_map (CouplingMap): Directed graph representing a coupling map.
            strict_direction (bool): If True, considers the direction of the coupling map.
                                     Default is False.
            seed (int): Sets the seed of the PRNG. -1 Means no node shuffling.
            call_limit (int): The number of state visits to attempt in each execution of
                VF2.
            time_limit (float): The total time limit in seconds to run ``VF2Layout``
            max_trials (int): The maximum number of trials to run VF2 to find
                a layout. If this is not specified the number of trials will be limited
                based on the number of edges in the interaction graph or the coupling graph
                (whichever is larger) if no other limits are set. If set to a value <= 0 no
                limit on the number of trials will be set.
            target (Target): A target representing the backend device to run ``VF2Layout`` on.
                If specified it will supersede a set value for
                ``coupling_map`` if the :class:`.Target` contains connectivity constraints. If the value
                of ``target`` models an ideal backend without any constraints then the value of
                ``coupling_map``
                will be used.

        Raises:
            TypeError: At runtime, if neither ``coupling_map`` or ``target`` are provided.
        """
        super().__init__()
        self.target = target
        self.coupling_map = coupling_map
        self.strict_direction = strict_direction
        self.seed = seed
        self.call_limit = call_limit
        self.time_limit = time_limit
        self.max_trials = max_trials
        self.avg_error_map = None

    def run(self, dag):
        """run the layout method"""
        if self.target is None and self.coupling_map is None:
            raise TranspilerError("coupling_map or target must be specified.")
        if self.coupling_map is None:
            target, coupling_map = self.target, self.target.build_coupling_map()
        elif self.target is None:
            coupling_map = self.coupling_map
            target = vf2_utils.build_dummy_target(coupling_map)
        else:
            # We have both, but may need to override the target if it has no connectivity.
            coupling_map = self.target.build_coupling_map()
            if coupling_map is None:
                target = vf2_utils.build_dummy_target(self.coupling_map)
                coupling_map = self.coupling_map
            else:
                target = self.target
        self.avg_error_map = self.property_set["vf2_avg_error_map"]
        if self.seed == -1:
            # Happy path of no shuffling.
            seed = None
        elif self.seed is None or self.seed < -1:
            # `seed is None` is OS entropy, `seed < -1` is a bad value (most pRNGs we're
            # concerned with deal with `u64`), but the stdlib `random` handles it, so just use
            # that to fix it up into something else.
            seed = random.Random(self.seed).randrange((1 << 64) - 1)
        else:
            seed = self.seed
        try:
            layout = vf2_layout_pass(
                dag,
                target,
                strict_direction=self.strict_direction,
                call_limit=self.call_limit,
                time_limit=self.time_limit,
                max_trials=self.max_trials,
                avg_error_map=self.avg_error_map,
                shuffle_seed=seed,
            )
        except MultiQEncountered:
            self.property_set["VF2Layout_stop_reason"] = VF2LayoutStopReason.MORE_THAN_2Q
            return
        if layout is None:
            self.property_set["VF2Layout_stop_reason"] = VF2LayoutStopReason.NO_SOLUTION_FOUND
            return

        self.property_set["VF2Layout_stop_reason"] = VF2LayoutStopReason.SOLUTION_FOUND
        layout = Layout({dag.qubits[virt]: phys for virt, phys in layout.items()})
        for reg in dag.qregs.values():
            layout.add_register(reg)
        self.property_set["layout"] = layout
