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

"""The Suzuki-Trotter product formula."""

from typing import Callable, Optional, Union

import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info.operators import SparsePauliOp, Pauli


from .product_formula import ProductFormula


class SuzukiTrotter(ProductFormula):
    r"""The (higher order) Suzuki-Trotter product formula.

    The Suzuki-Trotter formulas improve the error of the Lie-Trotter approximation.
    For example, the second order decomposition is

    .. math::

        e^{A + B} \approx e^{B/2} e^{A} e^{B/2}.

    Higher order decompositions are based on recursions, see Ref. [1] for more details.

    In this implementation, the operators are provided as sum terms of a Pauli operator.
    For example, in the second order Suzuki-Trotter decomposition we approximate

    .. math::

        e^{-it(XX + ZZ)} = e^{-it/2 ZZ}e^{-it XX}e^{-it/2 ZZ} + \mathcal{O}(t^3).

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
        order: int = 2,
        reps: int = 1,
        insert_barriers: bool = False,
        cx_structure: str = "chain",
        atomic_evolution: Optional[
            Callable[[Union[Pauli, SparsePauliOp], float], QuantumCircuit]
        ] = None,
    ) -> None:
        """
        Args:
            order: The order of the product formula.
            reps: The number of time steps.
            insert_barriers: Whether to insert barriers between the atomic evolutions.
            cx_structure: How to arrange the CX gates for the Pauli evolutions, can be "chain",
                where next neighbor connections are used, or "fountain", where all qubits are
                connected to one.
            atomic_evolution: A function to construct the circuit for the evolution of single
                Pauli string. Per default, a single Pauli evolution is decomposed in a CX chain
                and a single qubit Z rotation.
        Raises:
            ValueError: If order is not even
        """

        if order % 2 == 1:
            raise ValueError(
                "Suzuki product formulae are symmetric and therefore only defined "
                "for even orders."
            )
        super().__init__(order, reps, insert_barriers, cx_structure, atomic_evolution)

    def synthesize(self, evolution):
        # get operators and time to evolve
        operators = evolution.operator
        time = evolution.time

        if not isinstance(operators, list):
            pauli_list = [(Pauli(op), np.real(coeff)) for op, coeff in operators.to_list()]
        else:
            pauli_list = [(op, 1) for op in operators]

        ops_to_evolve = self._recurse(self.order, time / self.reps, pauli_list)

        # construct the evolution circuit
        single_rep = QuantumCircuit(operators[0].num_qubits)
        first_barrier = False

        for op, coeff in ops_to_evolve:
            # add barriers
            if first_barrier:
                if self.insert_barriers:
                    single_rep.barrier()
            else:
                first_barrier = True

            single_rep.compose(self.atomic_evolution(op, coeff), wrap=True, inplace=True)

        evolution_circuit = QuantumCircuit(operators[0].num_qubits)
        first_barrier = False

        for _ in range(self.reps):
            # add barriers
            if first_barrier:
                if self.insert_barriers:
                    single_rep.barrier()
            else:
                first_barrier = True

            evolution_circuit.compose(single_rep, inplace=True)

        return evolution_circuit

    @staticmethod
    def _recurse(order, time, pauli_list):
        if order == 1:
            return pauli_list

        elif order == 2:
            halves = [(op, coeff * time / 2) for op, coeff in pauli_list[:-1]]
            full = [(pauli_list[-1][0], time * pauli_list[-1][1])]
            return halves + full + list(reversed(halves))

        else:
            reduction = 1 / (4 - 4 ** (1 / (order - 1)))
            outer = 2 * SuzukiTrotter._recurse(
                order - 2, time=reduction * time, pauli_list=pauli_list
            )
            inner = SuzukiTrotter._recurse(
                order - 2, time=(1 - 4 * reduction) * time, pauli_list=pauli_list
            )
            return outer + inner + outer
