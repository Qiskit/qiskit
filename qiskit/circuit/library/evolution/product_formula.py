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

"""A gate to implement time-evolution of a single Pauli string."""


from typing import Callable, Optional, Union
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Pauli

from .evolution_synthesis import EvolutionSynthesis


class ProductFormula(EvolutionSynthesis):
    """Product formula base class for the decomposition of non-commuting operator exponentials.

    Lie-Trotter and Suzuki inherit this class.
    """

    def __init__(
        self,
        order: int,
        reps: int = 1,
        atomic_evolution: Optional[
            Callable[[Union[Pauli, SparsePauliOp, float]], QuantumCircuit]
        ] = None,
        insert_barriers: bool = False,
    ) -> None:
        """
        Args:
            order: The order of the product formula.
            reps: The number of time steps.
            atomic_evolution: A function to construct the circuit for the evolution of single operators.
                Per default, `PauliEvolutionGate` will be used.
            insert_barriers: Whether to insert barriers between the atomic evolutions.
        """
        super().__init__()
        self.order = order
        self.reps = reps
        self.insert_barriers = insert_barriers

        if atomic_evolution is None:
            from .pauli_evolution import PauliEvolutionGate

            def atomic_evolution(operator, time):
                evo = QuantumCircuit(operator.num_qubits)

                if isinstance(operator, Pauli):
                    # single Pauli operator: just exponentiate it
                    evo.append(PauliEvolutionGate(operator, time), evo.qubits)
                else:
                    # sum of Pauli operators: exponentiate each term (this assumes they commute)
                    pauli_list = [(Pauli(op), coeff) for op, coeff in operator.to_list()]
                    for pauli, coeff in pauli_list:
                        evo.append(PauliEvolutionGate(pauli, coeff * time), evo.qubits)

                return evo

        self.atomic_evolution = atomic_evolution
