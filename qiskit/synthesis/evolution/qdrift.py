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

from typing import Union, Optional, Callable
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info.operators import SparsePauliOp, Pauli

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

    def __init__(
        self,
        reps: int = 1,
        insert_barriers: bool = False,
        cx_structure: str = "chain",
        atomic_evolution: Optional[
            Callable[[Union[Pauli, SparsePauliOp], float], QuantumCircuit]
        ] = None,
        seed: Optional[int] = None,
    ) -> None:
        r"""
        Args:
            reps: The number of times to repeat the Trotterization circuit.
            insert_barriers: Whether to insert barriers between the atomic evolutions.
            cx_structure: How to arrange the CX gates for the Pauli evolutions, can be
                "chain", where next neighbor connections are used, or "fountain", where all
                qubits are connected to one.
            atomic_evolution: A function to construct the circuit for the evolution of single
                Pauli string. Per default, a single Pauli evolution is decomposed in a CX chain
                and a single qubit Z rotation.
            seed: An optional seed for reproducibility of the random sampling process.
        """
        super().__init__(1, reps, insert_barriers, cx_structure, atomic_evolution)
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

        num_gates = int(np.ceil(2 * (lambd**2) * (time**2) * self.reps))
        # The protocol calls for the removal of the individual coefficients,
        # and multiplication by a constant evolution time.
        evolution_time = lambd * time / num_gates

        self.sampled_ops = self.rng.choice(
            np.array(pauli_list, dtype=object),
            size=(num_gates,),
            p=weights / lambd,
        )
        # Update the coefficients of sampled_ops
        self.sampled_ops = [(op, evolution_time) for op, coeff in self.sampled_ops]

        # pylint: disable=cyclic-import
        from qiskit.circuit.library.pauli_evolution import PauliEvolutionGate

        # Build the evolution circuit using the LieTrotter synthesis with the sampled operators
        lie_trotter = LieTrotter(
            insert_barriers=self.insert_barriers, atomic_evolution=self.atomic_evolution
        )
        evolution_circuit = PauliEvolutionGate(
            sum(SparsePauliOp(op) for op, coeff in self.sampled_ops),
            time=evolution_time,
            synthesis=lie_trotter,
        ).definition

        return evolution_circuit
