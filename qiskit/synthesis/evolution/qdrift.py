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

"""QDrift Class"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info.operators import SparsePauliOp, Pauli
from qiskit.utils.deprecation import deprecate_arg

from .product_formula import ProductFormula
from .lie_trotter import LieTrotter


class QDrift(ProductFormula):
    r"""The QDrift Trotterization method, which selects each each term in the
    Trotterization randomly, with a probability proportional to its weight. Based on the work
    of Earl Campbell in Ref. [1].

    References:
        [1]: E. Campbell, "A random compiler for fast Hamiltonian simulation" (2018).
        `arXiv:quant-ph/1811.08017 <https://arxiv.org/abs/1811.08017>`_
    """

    @deprecate_arg(
        name="atomic_evolution",
        since="1.2",
        predicate=lambda callable: callable is not None
        and len(inspect.signature(callable).parameters) == 2,
        deprecation_description=(
            "The 'Callable[[Pauli | SparsePauliOp, float], QuantumCircuit]' signature of the "
            "'atomic_evolution' argument"
        ),
        additional_msg=(
            "Instead you should update your 'atomic_evolution' function to be of the following "
            "type: 'Callable[[QuantumCircuit, Pauli | SparsePauliOp, float], None]'."
        ),
        pending=True,
    )
    def __init__(
        self,
        reps: int = 1,
        insert_barriers: bool = False,
        cx_structure: str = "chain",
        atomic_evolution: (
            Callable[[Pauli | SparsePauliOp, float], QuantumCircuit]
            | Callable[[QuantumCircuit, Pauli | SparsePauliOp, float], None]
            | None
        ) = None,
        seed: int | None = None,
        wrap: bool = False,
    ) -> None:
        r"""
        Args:
            reps: The number of times to repeat the Trotterization circuit.
            insert_barriers: Whether to insert barriers between the atomic evolutions.
            cx_structure: How to arrange the CX gates for the Pauli evolutions, can be
                ``"chain"``, where next neighbor connections are used, or ``"fountain"``, where all
                qubits are connected to one. This only takes effect when
                ``atomic_evolution is None``.
            atomic_evolution: A function to apply the evolution of a single :class:`.Pauli`, or
                :class:`.SparsePauliOp` of only commuting terms, to a circuit. The function takes in
                three arguments: the circuit to append the evolution to, the Pauli operator to
                evolve, and the evolution time. By default, a single Pauli evolution is decomposed
                into a chain of ``CX`` gates and a single ``RZ`` gate.
                Alternatively, the function can also take Pauli operator and evolution time as
                inputs and returns the circuit that will be appended to the overall circuit being
                built.
            seed: An optional seed for reproducibility of the random sampling process.
            wrap: Whether to wrap the atomic evolutions into custom gate objects. This only takes
                effect when ``atomic_evolution is None``.
        """
        super().__init__(1, reps, insert_barriers, cx_structure, atomic_evolution, wrap)
        self.sampled_ops = None
        self.rng = np.random.default_rng(seed)

    def synthesize(self, evolution):
        # get operators and time to evolve
        operators = evolution.operator
        time = evolution.time

        if not isinstance(operators, list):
            pauli_list = [(Pauli(op), coeff) for op, coeff in operators.to_list()]
            coeffs = [np.real(coeff) for op, coeff in operators.to_list()]
        else:
            pauli_list = [(op, 1) for op in operators]
            coeffs = [1 for op in operators]

        # We artificially make the weights positive
        weights = np.abs(coeffs)
        lambd = np.sum(weights)

        num_gates = math.ceil(2 * (lambd**2) * (time**2) * self.reps)
        # The protocol calls for the removal of the individual coefficients,
        # and multiplication by a constant evolution time.
        evolution_time = lambd * time / num_gates

        self.sampled_ops = self.rng.choice(
            np.array(pauli_list, dtype=object),
            size=(num_gates,),
            p=weights / lambd,
        )

        # pylint: disable=cyclic-import
        from qiskit.circuit.library.pauli_evolution import PauliEvolutionGate

        # Build the evolution circuit using the LieTrotter synthesis with the sampled operators
        lie_trotter = LieTrotter(
            insert_barriers=self.insert_barriers, atomic_evolution=self.atomic_evolution
        )
        evolution_circuit = PauliEvolutionGate(
            sum(SparsePauliOp(np.sign(coeff) * op) for op, coeff in self.sampled_ops),
            time=evolution_time,
            synthesis=lie_trotter,
        ).definition

        return evolution_circuit
