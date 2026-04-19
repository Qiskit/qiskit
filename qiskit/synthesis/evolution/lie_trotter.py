# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Lie-Trotter product formula."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info.operators import SparsePauliOp
import qiskit.quantum_info

from .product_formula import real_or_fail
from .suzuki_trotter import SuzukiTrotter


@dataclass(frozen=True)
class _SpecialZOnlyCubicMatch:
    """Canonical representation of the narrow Z-only `{2, 2, 3}` pattern."""

    pivot: int
    leaf_a: int
    leaf_b: int
    coeff_leaf_a_pivot: float | Any
    coeff_pivot_leaf_b: float | Any
    coeff_three_body: float | Any


def _match_special_z_only_cubic_pattern(operator: object) -> _SpecialZOnlyCubicMatch | None:
    """Return a match for the narrow Z-only `{2, 2, 3}` cubic pattern if present."""
    if isinstance(operator, list):
        return None

    simplify = getattr(operator, "simplify", None)
    if callable(simplify):
        operator = simplify()

    to_sparse_list = getattr(operator, "to_sparse_list", None)
    num_qubits = getattr(operator, "num_qubits", None)
    if not callable(to_sparse_list) or not isinstance(num_qubits, int):
        return None

    terms = list(to_sparse_list())
    if len(terms) != 3:
        return None

    parsed_terms: list[tuple[tuple[int, ...], float | Any]] = []
    for pauli, qubits, coeff in terms:
        if len(pauli) != len(qubits) or set(pauli) != {"Z"}:
            return None

        try:
            real_coeff = real_or_fail(coeff)
        except ValueError:
            return None

        support = tuple(sorted(int(qubit) for qubit in qubits))
        parsed_terms.append((support, real_coeff))

    two_local = [(support, coeff) for support, coeff in parsed_terms if len(support) == 2]
    three_local = [(support, coeff) for support, coeff in parsed_terms if len(support) == 3]
    if len(two_local) != 2 or len(three_local) != 1:
        return None

    (support_a, coeff_a), (support_b, coeff_b) = two_local
    (support_three, coeff_three) = three_local[0]
    set_a = set(support_a)
    set_b = set(support_b)
    shared = set_a & set_b
    support_union = set_a | set_b
    if len(shared) != 1 or support_union != set(support_three):
        return None

    pivot = next(iter(shared))
    leaf_a, leaf_b = sorted(support_union - {pivot})

    coeff_leaf_a_pivot = coeff_a
    coeff_pivot_leaf_b = coeff_b
    if set(support_a) == {pivot, leaf_b} and set(support_b) == {leaf_a, pivot}:
        coeff_leaf_a_pivot, coeff_pivot_leaf_b = coeff_b, coeff_a
    elif set(support_a) != {leaf_a, pivot} or set(support_b) != {pivot, leaf_b}:
        return None

    return _SpecialZOnlyCubicMatch(
        pivot=pivot,
        leaf_a=leaf_a,
        leaf_b=leaf_b,
        coeff_leaf_a_pivot=coeff_leaf_a_pivot,
        coeff_pivot_leaf_b=coeff_pivot_leaf_b,
        coeff_three_body=coeff_three,
    )


def _synthesize_special_z_only_cubic(
    num_qubits: int, time: Any, match: _SpecialZOnlyCubicMatch
) -> QuantumCircuit:
    """Emit the exact `4 CX + 3 RZ` template for the narrow matched pattern."""
    circuit = QuantumCircuit(num_qubits)

    circuit.cx(match.leaf_a, match.pivot)
    circuit.rz(2 * match.coeff_leaf_a_pivot * time, match.pivot)

    circuit.cx(match.leaf_b, match.pivot)
    circuit.rz(2 * match.coeff_three_body * time, match.pivot)

    circuit.cx(match.leaf_a, match.pivot)
    circuit.rz(2 * match.coeff_pivot_leaf_b * time, match.pivot)

    circuit.cx(match.leaf_b, match.pivot)
    return circuit


class LieTrotter(SuzukiTrotter):
    r"""The Lie-Trotter product formula.

    The Lie-Trotter formula approximates the exponential of two non-commuting operators
    with products of their exponentials up to a second order error:

    .. math::

        e^{A + B} \approx e^{A}e^{B}.

    In this implementation, the operators are provided as sum terms of a Pauli operator.
    For example, we approximate

    .. math::

        e^{-it(XI + ZZ)} = e^{-it XI}e^{-it ZZ} + \mathcal{O}(t^2).

    References:

        [1]: D. Berry, G. Ahokas, R. Cleve and B. Sanders,
        "Efficient quantum algorithms for simulating sparse Hamiltonians" (2006).
        `arXiv:quant-ph/0508139 <https://arxiv.org/abs/quant-ph/0508139>`_
        [2]: N. Hatano and M. Suzuki,
        "Finding Exponential Product Formulas of Higher Orders" (2005).
        `arXiv:math-ph/0506007 <https://arxiv.org/pdf/math-ph/0506007.pdf>`_
    """

    def __init__(
        self,
        reps: int = 1,
        insert_barriers: bool = False,
        cx_structure: str = "chain",
        atomic_evolution: (
            Callable[[QuantumCircuit, qiskit.quantum_info.Pauli | SparsePauliOp, float], None]
            | None
        ) = None,
        wrap: bool = False,
        preserve_order: bool = True,
        *,
        atomic_evolution_sparse_observable: bool = False,
    ) -> None:
        r"""
        Args:
            reps: The number of time steps.
            insert_barriers: Whether to insert barriers between the atomic evolutions.
            cx_structure: How to arrange the CX gates for the Pauli evolutions, can be
                ``"chain"``, where next neighbor connections are used, or ``"fountain"``,
                where all qubits are connected to one. This only takes effect when
                ``atomic_evolution is None``.
            atomic_evolution: A function to apply the evolution of a single
                :class:`~.quantum_info.Pauli`, or :class:`.SparsePauliOp` of only commuting terms,
                to a circuit. The function takes in three arguments: the circuit to append the
                evolution to, the Pauli operator to evolve, and the evolution time. By default, a
                single Pauli evolution is decomposed into a chain of ``CX`` gates and a single
                ``RZ`` gate.
            wrap: Whether to wrap the atomic evolutions into custom gate objects. This only takes
                effect when ``atomic_evolution is None``.
            preserve_order: If ``False``, allows reordering the terms of the operator to
                potentially yield a shallower evolution circuit. Not relevant
                when synthesizing operator with a single term.
            atomic_evolution_sparse_observable: If a custom ``atomic_evolution`` is passed,
                which does not yet support :class:`.SparseObservable`\ s as input, set this
                argument to ``False`` to automatically apply a conversion to :class:`.SparsePauliOp`.
                This argument is supported until Qiskit 2.2, at which point all atomic evolutions
                are required to support :class:`.SparseObservable`\ s as input.
        """
        super().__init__(
            1,
            reps,
            insert_barriers,
            cx_structure,
            atomic_evolution,
            wrap,
            preserve_order=preserve_order,
            atomic_evolution_sparse_observable=atomic_evolution_sparse_observable,
        )

    @property
    def settings(self) -> dict[str, Any]:
        """Return the settings in a dictionary, which can be used to reconstruct the object.

        Returns:
            A dictionary containing the settings of this product formula.

        Raises:
            NotImplementedError: If a custom atomic evolution is set, which cannot be serialized.
        """
        if self._atomic_evolution is not None:
            raise NotImplementedError(
                "Cannot serialize a product formula with a custom atomic evolution."
            )

        return {
            "reps": self.reps,
            "insert_barriers": self.insert_barriers,
            "cx_structure": self._cx_structure,
            "wrap": self._wrap,
        }

    def synthesize(self, evolution):
        """Synthesize a :class:`.PauliEvolutionGate` with a narrow fast path for `#13285`."""
        if (
            self.reps == 1
            and self._atomic_evolution is None
            and not self._wrap
            and not self.insert_barriers
            and self._cx_structure == "chain"
            and not isinstance(evolution.operator, list)
        ):
            match = _match_special_z_only_cubic_pattern(evolution.operator)
            if match is not None:
                return _synthesize_special_z_only_cubic(evolution.num_qubits, evolution.time, match)

        return super().synthesize(evolution)
