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

"""A product formula base for decomposing non-commuting operator exponentials."""

from typing import Callable, Optional, Union
from functools import partial
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Pauli

from .evolution_synthesis import EvolutionSynthesis
from .pauli_evolution import evolve_pauli


class ProductFormula(EvolutionSynthesis):
    """Product formula base class for the decomposition of non-commuting operator exponentials.

    Lie-Trotter and Suzuki inherit this class.
    """

    def __init__(
        self,
        order: int,
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
            cx_structure: How to arrange the CX gates for the Pauli evolutions, can be
                "chain", where next neighbor connections are used, or "fountain", where all
                qubits are connected to one.
            atomic_evolution: A function to construct the circuit for the evolution of single
                Pauli string. Per default, a single Pauli evolution is decomopsed in a CX chain
                and a single qubit Z rotation.
        """
        super().__init__()
        self.order = order
        self.reps = reps
        self.insert_barriers = insert_barriers

        if atomic_evolution is None:
            atomic_evolution = partial(_default_atomic_evolution, cx_structure=cx_structure)

        self.atomic_evolution = atomic_evolution


def _default_atomic_evolution(operator, time, cx_structure):
    if isinstance(operator, Pauli):
        # single Pauli operator: just exponentiate it
        evo = evolve_pauli(operator, time, cx_structure)
    else:
        # sum of Pauli operators: exponentiate each term (this assumes they commute)
        pauli_list = [(Pauli(op), np.real(coeff)) for op, coeff in operator.to_list()]
        name = f"exp(it {[pauli.to_label() for pauli, _ in pauli_list]})"
        evo = QuantumCircuit(operator.num_qubits, name=name)
        for pauli, coeff in pauli_list:
            evo.compose(evolve_pauli(pauli, coeff * time, cx_structure), inplace=True)

    return evo
