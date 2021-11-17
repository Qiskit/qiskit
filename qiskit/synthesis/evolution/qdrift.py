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

from typing import List, Union, Optional, Callable, Tuple
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info.operators import SparsePauliOp, Pauli
from qiskit.utils import algorithm_globals

from .product_formula import ProductFormula


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
    ) -> None:
        r"""
        questions: order 1 the trotterisation. Should i make it as an argument in init or directly pass 1 to super?
        Args:
            reps: The number of times to repeat the Trotterization circuit.
        """
        super().__init__(1, reps, insert_barriers, cx_structure, atomic_evolution)
        self.sampled_ops = None

    @property
    def sampled_ops(self) -> List[Tuple[Pauli, float]]:
        """returns the list of sampled Pauli ops and their coefficients"""
        return self._sampled_ops

    @sampled_ops.setter
    def sampled_ops(self, sampled_ops: List[Tuple[Pauli, float]]) -> None:
        """sets the list of sampled Pauli ops and their coefficients"""
        self._sampled_ops = sampled_ops

    def synthesize(self, evolution):
        # get operators and time to evolve
        operators = evolution.operator
        time = evolution.time

        if not isinstance(operators, list):
            pauli_list = [(Pauli(op), np.real(coeff)) for op, coeff in operators.to_list()]
            coeffs = [np.real(coeff) for op, coeff in operators.to_list()]
        else:
            pauli_list = [(op, 1) for op in operators]
            coeffs = [1 for op in operators]

        # We artificially make the weights positive, TODO check approximation performance
        weights = np.abs(coeffs)
        lambd = np.sum(weights)

        N = 2 * (lambd ** 2) * (time ** 2)
        factor = lambd * time / N * self.reps
        # The protocol calls for the removal of the individual coefficients,
        # and multiplication by a constant factor.
        scaled_ops = [(op, factor / coeff) for op, coeff in pauli_list]
        self.sampled_ops = algorithm_globals.random.choice(np.array(scaled_ops, dtype=object),
                                                           size=(int(np.ceil(N * self.reps)),),
                                                           p=weights / lambd)

        # construct the evolution circuit
        evo = QuantumCircuit(operators[0].num_qubits)
        first_barrier = False

        for op, coeff in self.sampled_ops:
            # add barriers
            if first_barrier:
                if self.insert_barriers:
                    evo.barrier()
            else:
                first_barrier = True

            evo.compose(self.atomic_evolution(op, coeff), wrap=True, inplace=True)

        return evo
