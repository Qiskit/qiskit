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

"""The Lie-Trotter product formula."""

from typing import List, Callable, Optional, Union
import numpy as np
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info.operators import SparsePauliOp, Pauli

from .product_formula import ProductFormula


class LieTrotter(ProductFormula):
    """The Lie-Trotter product formula.

    References:

        [1]: D. Berry, G. Ahokas, R. Cleve and B. Sanders,
        "Efficient quantum algorithms for simulating sparse Hamiltonians" (2006).
        `arXiv:quant-ph/0508139 <https://arxiv.org/abs/quant-ph/0508139>`_
    """

    def __init__(
        self,
        reps: int = 1,
        insert_barriers: bool = False,
        cx_structure: str = "chain",
        atomic_evolution: Optional[
            Callable[[Union[Pauli, SparsePauliOp], float], QuantumCircuit]
        ] = None,
    ) -> None:
        """
        Args:
            reps: The number of time steps.
            insert_barriers: If True, insert barriers in between each evolved Pauli.
            cx_structure: How to arrange the CX gates for the Pauli evolutions, can be
                "chain", where next neighbor connections are used, or "fountain", where all
                qubits are connected to one.
            atomic_evolution: A function to construct the circuit for the evolution of single operators.
                Per default, `PauliEvolutionGate` will be used.
        """
        super().__init__(1, reps, insert_barriers, cx_structure, atomic_evolution)

    def synthesize(
        self,
        operators: Union[SparsePauliOp, List[SparsePauliOp]],
        time: Union[float, ParameterExpression],
    ) -> QuantumCircuit:
        evo = QuantumCircuit(operators[0].num_qubits)
        first_barrier = False

        if not isinstance(operators, list):
            pauli_list = [(Pauli(op), np.real(coeff)) for op, coeff in operators.to_list()]
        else:
            pauli_list = [(op, 1) for op in operators]

        for _ in range(self.reps):
            for op, coeff in pauli_list:
                # add barriers
                if first_barrier:
                    if self.insert_barriers:
                        evo.barrier()
                else:
                    first_barrier = True

                evo.compose(self.atomic_evolution(op, coeff * time / self.reps), inplace=True)

        return evo
