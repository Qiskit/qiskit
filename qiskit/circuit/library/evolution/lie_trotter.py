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

from typing import List, Callable, Optional
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info.operators import SparsePauliOp

from .product_formula import ProductFormula


class LieTrotter(ProductFormula):
    """The Lie-Trotter product formula."""

    def __init__(self,
                 reps: int = 1,
                 atomic_evolution: Optional[Callable[[SparsePauliOp, float], QuantumCircuit]] = None
                 ) -> None:
        """
        Args:
            order: The order of the product formula.
            reps: The number of time steps.
            atomic_evolution: A function to construct the circuit for the evolution of single operators.
                Per default, `PauliEvolutionGate` will be used.
        """
        super().__init__(order=1, reps=reps, atomic_evolution=atomic_evolution)

    def synthesize(self, operators: List[SparsePauliOp], time: float) -> QuantumCircuit:
        # TODO move logic here
        from qiskit.opflow import PauliTrotterEvolution, PauliSumOp
        op = sum(operators[1:], operators[0])
        exp = (time * PauliSumOp(op)).exp_i()
        return PauliTrotterEvolution(reps=self.reps).convert(exp).to_circuit_op().primitive
