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
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info.operators import SparsePauliOp

from .product_formula import ProductFormula


class LieTrotter(ProductFormula):
    """The Lie-Trotter product formula."""

    def __init__(
        self,
        reps: int = 1,
        atomic_evolution: Optional[Callable[[SparsePauliOp, float], QuantumCircuit]] = None,
        insert_barriers: bool = False,
    ) -> None:
        """
        Args:
            order: The order of the product formula.
            reps: The number of time steps.
            atomic_evolution: A function to construct the circuit for the evolution of single operators.
                Per default, `PauliEvolutionGate` will be used.
        """
        super().__init__(1, reps, atomic_evolution, insert_barriers)

    def synthesize(
        self, operators: Union[SparsePauliOp, List[SparsePauliOp]], time: float
    ) -> QuantumCircuit:
        evo = QuantumCircuit(operators[0].num_qubits)
        first_barrier = False
        for _ in range(self.reps):
            for op in operators:
                # add barriers
                if first_barrier:
                    if self.insert_barriers:
                        evo.barrier()
                else:
                    first_barrier = True

                evo.compose(self.atomic_evolution(op, time / self.reps), inplace=True)

        return evo
