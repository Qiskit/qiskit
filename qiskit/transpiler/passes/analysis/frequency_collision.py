# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Collision analysis."""

import enum
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

from qiskit.dagcircuit import DAGCircuit
from qiskit.providers.models import BackendProperties
from qiskit.transpiler import Target, CouplingMap
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError


@dataclass
class Collision:
    """
    TODO: Please write the docstring here
    """

    control: int
    """The control qubit that this frequency collision occurs on."""
    target: int
    """The target qubit that this frequency collision occurs on."""
    spectator: Optional[int]
    """The spectator qubit that this frequency collision occurs on."""
    lhs: float
    """A sequence of the classical bits that this operation reads from or writes to."""


class CollisionTypes(enum.Enum):
    """
    TODO: Please write the docstring here
    """

    TYPE1 = "F01_F01"
    TYPE2 = "CR_F02/2_F01"
    TYPE3 = "F01_F12"
    TYPE4 = "SLOW_CR"
    TYPE5 = "SPECTATOR_F01_F01"
    TYPE6 = "SPECTATOR_F01_F12"
    TYPE7 = "SPECTATOR_F02/2_F01"


class CollisionAnalysis(AnalysisPass):
    """TODO: Please write the docstring here

    - Mention to consider only native directions.
    - Add info of the reference papaer
    etc.
    """

    __frq_str = "frequency"
    __anh_str = "anharmonicity"

    def __init__(
        self,
        target: Target = None,
        native_cmap: Optional[List[Tuple[int, int]]] = None,
        type1_bound: Optional[Tuple[float, float]] = None,
        type2_bound: Optional[Tuple[float, float]] = None,
        type3_bound: Optional[Tuple[float, float]] = None,
        type5_bound: Optional[Tuple[float, float]] = None,
        type6_bound: Optional[Tuple[float, float]] = None,
        type7_bound: Optional[Tuple[float, float]] = None,
        properties: Optional[BackendProperties] = None,
        coupling_map: Optional[CouplingMap] = None,
    ):
        """TODO: Please write the docstring here

        Args:
            target: A target representing the backend device to run ``CollisionAnalysis`` on.
                If specified, it will supersede a set value for ``properties`` and ``coupling_map``.
            native_camp: A list of the directions of cross-resonance pulses
                natively supported by the backend device to run ``CollisionAnalysis`` on.
                Each direction is given as a pair of control and target (physical) qubits.
                If not specified, it may be created from ``target`` or
                (``properties`` and ``coupling_map``).
            type1_bound: Margin from the exact Type 1 collision frequencies.
                It is a pair of lower and upper bounds in Hz, e.g. (-10e6, 20e6),
                which means -10MHz < (LHS - RHS) frequency < 20MHz.
                where LHS and RHS are left- and right-hand-side expression in the collision definition.
                If not specified, the value (-17e6, 17e6) in the reference paper will be used.
            type2_bound: Margin from the exact Type 2 collision frequencies.
                If not specified, the value (-4e6, 4e6) in the reference paper will be used.
            type3_bound: Margin from the exact Type 3 collision frequencies.
                If not specified, the value (-30e6, 30e6) in the reference paper will be used.
            type5_bound: Margin from the exact Type 4 collision frequencies.
                If not specified, the value (-17e6, 17e6) in the reference paper will be used.
            type6_bound: Margin from the exact Type 6 collision frequencies.
                If not specified, the value (-25e6, 25e6) in the reference paper will be used.
            type7_bound: Margin from the exact Type 7 collision frequencies.
                If not specified, the value (-17e6, 17e6) in the reference paper will be used.
            properties: A properties representing the backend device to run ``CollisionAnalysis`` on.
                Either of ``target`` or (``properties`` and ``coupling_map``) must be supplied.
            coupling_map: A coupling map of the backend device to run ``CollisionAnalysis`` on.

        Raises:
            TranspilerError: If neither ``target`` or ``properties`` are provided.
        """
        super().__init__()
        if target is None:
            if properties is None:
                raise TranspilerError("`target` or `properties` must be provided")
            if native_cmap is None:
                if properties is None or coupling_map is None:
                    raise TranspilerError(
                        "`properties` and `coupling_map` must be supplied "
                        "if neither `target` or `native_camp` are provided"
                    )

        self.target = target
        self.properties = properties

        self.native_cmap = native_cmap or []
        if not self.native_cmap:
            # Create native_cmap from target or (properties and coupling_map)
            coupling = coupling_map
            if self.target:
                coupling = self.target.build_coupling_map()

            for q0, q1 in coupling.get_edges():
                if q0 > q1:
                    continue
                cx01 = self._get_cx_gate_length(q0, q1)
                cx10 = self._get_cx_gate_length(q1, q0)
                if cx01 is None or cx10 is None:
                    raise TranspilerError(
                        "CollisionAnalysis requires backend reports duration of all cx gates "
                        "when `native_camp` is not provided"
                    )
                if cx01 > cx10:
                    self.native_cmap.append((q1, q0))
                else:
                    self.native_cmap.append((q0, q1))

        neighbor = defaultdict(set)
        for j, k in self.native_cmap:
            neighbor[j].add(k)
            neighbor[k].add(j)

        self._neighbor = dict(neighbor)

        self.type1_bound = type1_bound or (-17e6, 17e6)
        self.type2_bound = type2_bound or (-4e6, 4e6)
        self.type3_bound = type3_bound or (-30e6, 30e6)
        self.type5_bound = type5_bound or (-17e6, 17e6)
        self.type6_bound = type6_bound or (-25e6, 25e6)
        self.type7_bound = type7_bound or (-17e6, 17e6)

    def run(self, dag: DAGCircuit):
        self.property_set["collisions"] = {t: [] for t in CollisionTypes}
        # Two-qubit collisions
        for q_c, q_t in self.native_cmap:
            f01_c = self._get_frequency(q_c)
            f01_t = self._get_frequency(q_t)
            if f01_c is None or f01_t is None:
                raise TranspilerError(
                    "CollisionAnalysis requires backend reports frequencies of all physical qubits"
                )
            anh_c = self._get_anharmonicity(q_c)
            anh_t = self._get_anharmonicity(q_t)
            if anh_c is None or anh_t is None:
                raise TranspilerError(
                    "CollisionAnalysis requires backend reports anharmonicities of all physical qubits"
                )

            f12_c = f01_c + anh_c
            f12_t = f01_t + anh_t
            f02_c = 2 * f01_c + anh_c

            # Direct f01 collision, which induces non-negligible always-on coupling.
            self._add_constraint(
                collision_type=CollisionTypes.TYPE1,
                lhs=f01_c - f01_t,
                q_ctrl=q_c,
                q_tgt=q_t,
                lb=self.type1_bound[0],
                ub=self.type1_bound[1],
            )
            # CR may drive target qubit with high power.
            # If this frequency corresponds to f02/2 of the control qubit,
            # this CR drive may have sufficient power to directly drive f02 of the control.
            self._add_constraint(
                collision_type=CollisionTypes.TYPE2,
                lhs=f02_c - 2 * f01_t,
                lb=self.type2_bound[0],
                ub=self.type2_bound[1],
                q_ctrl=q_c,
                q_tgt=q_t,
            )
            # This is special case we need to consider both direction.
            # When we drive a single qubit gate on some qubit and its frequency corresponds to
            # the f12 of adjacent qubit, this accidentally drive CX12 gate in qutrit subspace.
            # This is significant when adjacent qubit state has finite probability in |1>.
            self._add_constraint(
                collision_type=CollisionTypes.TYPE3,
                lhs=f12_c - f01_t,
                q_ctrl=q_c,
                q_tgt=q_t,
                lb=self.type3_bound[0],
                ub=self.type3_bound[1],
            )
            self._add_constraint(
                collision_type=CollisionTypes.TYPE3,
                lhs=f12_t - f01_c,
                q_ctrl=q_t,
                q_tgt=q_c,
                lb=self.type3_bound[0],
                ub=self.type3_bound[1],
            )
            # This is not a collision, but preferred configuration to run CR efficiently.
            # Usually it is preferred to have higher control qubit frequency,
            # and CR qubits detuning is not too high, typically smaller than anharmonicity.
            cr_detuning = np.abs(f01_c - f01_t)
            self._add_constraint(
                collision_type=CollisionTypes.TYPE4,
                lhs=f01_c - f01_t,
                q_ctrl=q_c,
                q_tgt=q_t,
                ub=0,
            )
            self._add_constraint(
                collision_type=CollisionTypes.TYPE4,
                lhs=cr_detuning - np.abs(anh_c),
                q_ctrl=q_c,
                q_tgt=q_t,
                ub=0,
            )

            # Spectator collisions
            for q_s in self._neighbor[q_c] - {q_t}:
                f01_s = self._get_frequency(q_s)
                anh_s = self._get_anharmonicity(q_s)
                if f01_s is None or anh_s is None:
                    raise TranspilerError(
                        "CollisionAnalysis requires backend reports frequencies and anharmonicities"
                        " of all physical qubits"
                    )
                f12_s = f01_s + anh_s

                # This is spectator f01 collision.
                # When we drive CX[q_c, q_t] this may also drive CR with the spectator q_s.
                self._add_constraint(
                    collision_type=CollisionTypes.TYPE5,
                    lhs=f01_t - f01_s,
                    q_ctrl=q_c,
                    q_tgt=q_t,
                    q_spectator=q_s,
                    lb=self.type5_bound[0],
                    ub=self.type5_bound[1],
                )
                # This is spectator f01-f12 collision.
                # When we drive CX[q_c, q_t] this may also drive CR12 with the spectator q_s
                # in the qutrit subspace. This is significant when spectator state has
                # finite probability in |1>
                self._add_constraint(
                    collision_type=CollisionTypes.TYPE6,
                    lhs=f01_t - f12_s,
                    q_ctrl=q_c,
                    q_tgt=q_t,
                    q_spectator=q_s,
                    lb=self.type6_bound[0],
                    ub=self.type6_bound[1],
                )
                # This is control qubit leakage with level degeneration.
                # Define |jik> as a state of |q_c>|q_t>|q_s>.
                # This condition degenerate |200> and |011> and control qubit may leak to |2>.
                self._add_constraint(
                    collision_type=CollisionTypes.TYPE7,
                    lhs=f02_c - (f01_t + f01_s),
                    q_ctrl=q_c,
                    q_tgt=q_t,
                    q_spectator=q_s,
                    lb=self.type7_bound[0],
                    ub=self.type7_bound[1],
                )
        # Store the bound data additionally
        self.property_set["collision_bound"] = {
            CollisionTypes.TYPE1: self.type1_bound,
            CollisionTypes.TYPE2: self.type2_bound,
            CollisionTypes.TYPE3: self.type3_bound,
            CollisionTypes.TYPE5: self.type5_bound,
            CollisionTypes.TYPE6: self.type6_bound,
            CollisionTypes.TYPE7: self.type7_bound,
        }

    def _get_cx_gate_length(self, control: int, target: int) -> Optional[float]:
        try:
            if self.target:
                return self.target["cx"][(control, target)].duration
            return self.properties.gate_length("cx", (control, target))
        except KeyError:
            return None

    def _get_frequency(self, qubit: int) -> Optional[float]:
        if self.target:
            if self.target.qubit_properties:
                return self.target.qubit_properties[qubit].frequency
            return None
        else:
            try:
                return self.properties.qubit_property(qubit, self.__frq_str)[0]
            except KeyError:
                return None

    def _get_anharmonicity(self, qubit: int) -> Optional[float]:
        if self.target:
            if self.target.qubit_properties:
                return getattr(self.target.qubit_properties[qubit], self.__anh_str, None)
            return None
        else:
            try:
                return self.properties.qubit_property(qubit, self.__anh_str)[0]
            except KeyError:
                return None

    def _add_constraint(
        self,
        collision_type: CollisionTypes,
        lhs: float,
        q_ctrl: int,
        q_tgt: int,
        lb: Optional[float] = None,
        ub: Optional[float] = None,
        q_spectator: Optional[int] = None,
    ):
        violate = True
        if lb is not None and lhs < lb:
            violate = False
        if ub is not None and lhs > ub:
            violate = False
        if violate:
            self.property_set["collisions"][collision_type].append(
                Collision(control=q_ctrl, target=q_tgt, spectator=q_spectator, lhs=lhs)
            )
