# This code is part of Qiskit.
#
# (C) Copyright IBM 2021
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Driver for a synthesis routine which emits optimal ZX-based circuits.
"""

import heapq
import math
from operator import itemgetter
from typing import Callable, Optional

import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import RZXGate
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info.synthesis.one_qubit_decompose import ONE_QUBIT_EULER_BASIS_GATES
from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitWeylDecomposition
from qiskit.transpiler.passes.optimization.optimize_1q_decomposition import (
    Optimize1qGatesDecomposition,
)

from .circuits import apply_reflection, apply_shift, canonical_xx_circuit
from .utilities import EPSILON
from .polytopes import XXPolytope


def average_infidelity(p, q):
    """
    Computes the infidelity distance between two points p, q expressed in positive canonical
    coordinates.
    """

    a0, b0, c0 = p
    a1, b1, c1 = q

    return 1 - 1 / 20 * (
        4
        + 16
        * (
            math.cos(a0 - a1) ** 2 * math.cos(b0 - b1) ** 2 * math.cos(c0 - c1) ** 2
            + math.sin(a0 - a1) ** 2 * math.sin(b0 - b1) ** 2 * math.sin(c0 - c1) ** 2
        )
    )


class MonodromyZXDecomposer:
    """
    A class for optimal decomposition of 2-qubit unitaries into 2-qubit basis gates of XX type
    (i.e., each locally equivalent to CAN(alpha, 0, 0) for a possibly varying alpha).

    Args:
        euler_basis: Basis string provided to OneQubitEulerDecomposer for 1Q synthesis.
            Defaults to "U3".
        embodiments: An dictionary mapping interaction strengths alpha to native circuits which
            embody the gate CAN(alpha, 0, 0). Strengths are taken to be normalized, so that 1/2
            represents the class of a full CX.
        backup_optimizer: If supplied, defers synthesis to this callable when MonodromyZXDecomposer
            has no efficient decomposition of its own.

    NOTE: If embodiments is not passed, or if an entry is missing, it will be populated as needed
        using the method _default_embodiment.
    """

    def __init__(
        self,
        euler_basis: str = "U3",
        embodiments: Optional[dict] = None,
        backup_optimizer: Optional[Callable] = None,
    ):
        self._decomposer1q = Optimize1qGatesDecomposition(ONE_QUBIT_EULER_BASIS_GATES[euler_basis])
        self.gate = RZXGate(np.pi / 2)
        self.embodiments = embodiments if embodiments is not None else {}
        self.backup_optimizer = backup_optimizer

    @staticmethod
    def _default_embodiment(strength):
        """
        If the user does not provide a custom implementation of XX(strength), then this routine
        defines a default implementation using RZX or CX.
        """
        xx_circuit = QuantumCircuit(2)

        if strength == np.pi / 2:
            xx_circuit.h(0)
            xx_circuit.cx(0, 1)
            xx_circuit.h(1)
            xx_circuit.rz(np.pi / 2, 0)
            xx_circuit.rz(np.pi / 2, 1)
            xx_circuit.h(1)
            xx_circuit.h(0)
            xx_circuit.global_phase += np.pi / 4
        else:
            xx_circuit.h(0)
            xx_circuit.rzx(strength, 0, 1)
            xx_circuit.h(0)

        return xx_circuit

    @staticmethod
    def _best_decomposition(canonical_coordinate, available_strengths):
        """
        Finds the cheapest sequence of `available_strengths` which supports the best approximation
        to `canonical_coordinate`. Returns a dictionary with keys "cost", "point", and "operations".

        NOTE: `canonical_coordinate` is a positive canonical coordinate. `strengths` is a dictionary
              mapping the available strengths to their (infidelity) costs, with the strengths
              themselves normalized so that pi/2 represents CX = RZX(pi/2).
        """
        best_point, best_cost, best_sequence = [0, 0, 0], 1.0, []
        priority_queue = []
        heapq.heappush(priority_queue, (0, []))
        canonical_coordinate = np.array(canonical_coordinate)

        while True:
            sequence_cost, sequence = heapq.heappop(priority_queue)

            strength_polytope = XXPolytope.from_strengths(*[x / 2 for x in sequence])
            candidate_point = strength_polytope.nearest(canonical_coordinate)
            candidate_cost = sequence_cost + average_infidelity(
                canonical_coordinate, candidate_point
            )

            if candidate_cost < best_cost:
                best_point, best_cost, best_sequence = candidate_point, candidate_cost, sequence

            if strength_polytope.member(canonical_coordinate):
                break

            for strength, extra_cost in available_strengths.items():
                if len(sequence) == 0 or strength <= sequence[-1]:
                    heapq.heappush(
                        priority_queue, (sequence_cost + extra_cost, sequence + [strength])
                    )

        return {"point": best_point, "cost": best_cost, "sequence": best_sequence}

    def num_basis_gates(self, unitary):
        """
        Counts the number of gates that would be emitted during re-synthesis.

        NOTE: Used by ConsolidateBlocks.
        """
        strengths = self._strength_to_infidelity(1.0)

        # get the associated _positive_ canonical coordinate
        weyl_decomposition = TwoQubitWeylDecomposition(unitary)
        target = [getattr(weyl_decomposition, x) for x in ("a", "b", "c")]
        if target[-1] < -EPSILON:
            target = [np.pi / 2 - target[0], target[1], -target[2]]

        best_sequence = self._best_decomposition(target, strengths)["sequence"]
        return len(best_sequence)

    @staticmethod
    def _strength_to_infidelity(basis_fidelity, approximate=False):
        """
        Converts a dictionary mapping ZX strengths to fidelities to a dictionary mapping ZX
        strengths to infidelities. Also supports some of the other formats QISKit uses: injects a
        default set of infidelities for CX, CX/2, and CX/3 if None is supplied, or extends a single
        float infidelity over CX, CX/2, and CX/3 if only a single float is supplied.
        """

        if basis_fidelity is None or isinstance(basis_fidelity, float):
            if not approximate:
                slope, offset = 1e-10, 1e-12
            elif isinstance(basis_fidelity, float):
                slope, offset = (1 - basis_fidelity) / 2, (1 - basis_fidelity) / 2
            else:
                # some reasonable default values
                slope, offset = (64 * 90) / 1000000, 909 / 1000000 + 1 / 1000
            return {
                strength: slope * strength / (np.pi / 2) + offset
                for strength in [np.pi / 2, np.pi / 4, np.pi / 6]
            }
        elif isinstance(basis_fidelity, dict):
            return {
                strength: (1 - fidelity if approximate else 1e-12 + 1e-10 * strength / (np.pi / 2))
                for (strength, fidelity) in basis_fidelity.items()
            }

        raise TypeError("Unknown basis_fidelity payload.")

    def __call__(self, unitary, basis_fidelity=None, approximate=True, chatty=False):
        """
        Fashions a circuit which (perhaps `approximate`ly) models the special unitary operation `u`,
        using the circuit templates supplied at initialization.  The routine uses `basis_fidelity`
        to select the optimal circuit template, including when performing exact synthesis; the
        contents of `basis_fidelity` is a dictionary mapping interaction strengths (scaled so that
        CX = RZX(pi/2) corresponds to pi/2) to circuit fidelities.
        """
        strength_to_infidelity = self._strength_to_infidelity(
            basis_fidelity, approximate=approximate
        )

        # get the associated _positive_ canonical coordinate
        weyl_decomposition = TwoQubitWeylDecomposition(unitary)
        target = [getattr(weyl_decomposition, x) for x in ("a", "b", "c")]
        if target[-1] < -EPSILON:
            target = [np.pi / 2 - target[0], target[1], -target[2]]

        # scan for the best point
        best_point, best_sequence = itemgetter("point", "sequence")(
            self._best_decomposition(target, strength_to_infidelity)
        )
        # build the circuit building this canonical gate
        embodiments = {
            k: self.embodiments.get(k, self._default_embodiment(k))
            for k, v in strength_to_infidelity.items()
        }
        circuit = canonical_xx_circuit(best_point, best_sequence, embodiments)

        if best_sequence == [np.pi / 2, np.pi / 2, np.pi / 2] and self.backup_optimizer is not None:
            return self.backup_optimizer(unitary, basis_fidelity=basis_fidelity)

        # change to positive canonical coordinates
        if weyl_decomposition.c >= -EPSILON:
            # if they're the same...
            corrected_circuit = QuantumCircuit(2)
            corrected_circuit.rz(np.pi, [0])
            corrected_circuit.compose(circuit, [0, 1], inplace=True)
            corrected_circuit.rz(-np.pi, [0])
            circuit = corrected_circuit
        else:
            # else they're in the "positive" scissors part...
            corrected_circuit = QuantumCircuit(2)
            _, source_reflection, _ = apply_reflection("reflect XX, ZZ", [0, 0, 0])
            _, source_shift, _ = apply_shift("X shift", [0, 0, 0])

            corrected_circuit.compose(source_reflection.inverse(), inplace=True)
            corrected_circuit.rz(np.pi, [0])
            corrected_circuit.compose(circuit, [0, 1], inplace=True)
            corrected_circuit.rz(-np.pi, [0])
            corrected_circuit.compose(source_shift.inverse(), inplace=True)
            corrected_circuit.compose(source_reflection, inplace=True)
            corrected_circuit.global_phase += np.pi / 2

            circuit = corrected_circuit

        circ = QuantumCircuit(2, global_phase=weyl_decomposition.global_phase)

        circ.append(UnitaryGate(weyl_decomposition.K2r), [0])
        circ.append(UnitaryGate(weyl_decomposition.K2l), [1])
        circ.compose(circuit, [0, 1], inplace=True)
        circ.append(UnitaryGate(weyl_decomposition.K1r), [0])
        circ.append(UnitaryGate(weyl_decomposition.K1l), [1])

        circ = self._decomposer1q(circ)

        return circ
