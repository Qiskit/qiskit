# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An algorithm to implement a Trotterization real time-evolution."""

from typing import Union, Optional

from qiskit import QuantumCircuit
from qiskit.algorithms.observables_evaluator import eval_observables
from qiskit.algorithms.time_evolvers.evolution_problem import EvolutionProblem
from qiskit.algorithms.time_evolvers.evolution_result import EvolutionResult
from qiskit.algorithms.time_evolvers.real_evolver import RealEvolver
from qiskit.opflow import (
    SummedOp,
    PauliOp,
    CircuitOp,
    PauliSumOp,
    StateFn,
    OperatorBase,
)
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.primitives import Estimator
from qiskit.synthesis import ProductFormula, LieTrotter


class TrotterQRTE(RealEvolver):
    """Quantum Real Time Evolution using Trotterization.
    Type of Trotterization is defined by a ProductFormula provided.

    Examples:

        .. jupyter-execute::

            from qiskit.opflow import X, Z, Zero
            from qiskit.algorithms import EvolutionProblem, TrotterQRTE

            operator = X + Z
            initial_state = Zero
            time = 1
            evolution_problem = EvolutionProblem(operator, 1, initial_state)
            # LieTrotter with 1 rep
            trotter_qrte = TrotterQRTE()
            evolved_state = trotter_qrte.evolve(evolution_problem).evolved_state
    """

    def __init__(
        self,
        product_formula: Optional[ProductFormula] = None,
        estimator: Estimator = None,
    ) -> None:
        """
        Args:
            product_formula: A Lie-Trotter-Suzuki product formula. The default is the Lie-Trotter
                first order product formula with a single repetition.
            estimator: An estimator primitive used for calculating expectation values of
                EvolutionProblem.aux_operators.
        """
        if product_formula is None:
            product_formula = LieTrotter()
        self.product_formula = product_formula
        self.estimator = estimator

    @classmethod
    def supports_aux_operators(cls) -> bool:
        """
        Whether computing the expectation value of auxiliary operators is supported.

        Returns:
            True if ``aux_operators`` expectations in the EvolutionProblem can be evaluated, False
                otherwise.
        """
        return True

    def evolve(self, evolution_problem: EvolutionProblem) -> EvolutionResult:
        """
        Evolves a quantum state for a given time using the Trotterization method
        based on a product formula provided. The result is provided in the form of a quantum
        circuit. If auxiliary operators are included in the ``evolution_problem``, they are
        evaluated on an evolved state using an estimator primitive provided.

        .. note::
            Time-dependent Hamiltonians are not yet supported.

        Args:
            evolution_problem: Instance defining evolution problem. For the included Hamiltonian,
                ``PauliOp``, ``SummedOp`` or ``PauliSumOp`` are supported by TrotterQRTE.

        Returns:
            Evolution result that includes an evolved state as a quantum circuit and, optionally,
            auxiliary operators evaluated for a resulting state on an estimator primitive.

        Raises:
            ValueError: If ``t_param`` is not set to None in the EvolutionProblem (feature not
                currently supported).
            ValueError: If the ``initial_state`` is not provided in the EvolutionProblem.
        """
        evolution_problem.validate_params()
        if evolution_problem.t_param is not None:
            raise ValueError(
                "TrotterQRTE does not accept a time dependent hamiltonian,"
                "``t_param`` from the EvolutionProblem should be set to None."
            )

        if evolution_problem.aux_operators is not None and (self.estimator is None):
            raise ValueError(
                "aux_operators were provided for evaluations but no ``estimator`` was provided."
            )
        hamiltonian = evolution_problem.hamiltonian
        if not isinstance(hamiltonian, (PauliOp, PauliSumOp, SummedOp)):
            raise ValueError(
                "TrotterQRTE only accepts PauliOp | "
                f"PauliSumOp | SummedOp, {type(hamiltonian)} provided."
            )
        if isinstance(hamiltonian, OperatorBase):
            hamiltonian = hamiltonian.bind_parameters(evolution_problem.param_value_dict)
        if isinstance(hamiltonian, SummedOp):
            hamiltonian = self._summed_op_to_pauli_sum_op(hamiltonian)
        # the evolution gate
        evolution_gate = CircuitOp(
            PauliEvolutionGate(hamiltonian, evolution_problem.time, synthesis=self.product_formula)
        )

        if evolution_problem.initial_state is not None:
            initial_state = evolution_problem.initial_state
            if isinstance(initial_state, QuantumCircuit):
                initial_state = StateFn(initial_state)
            evolved_state = evolution_gate @ initial_state

        else:
            raise ValueError("``initial_state`` must be provided in the EvolutionProblem.")

        evaluated_aux_ops = None
        if evolution_problem.aux_operators is not None:
            evaluated_aux_ops = eval_observables(
                self.estimator,
                evolved_state.primitive,
                evolution_problem.aux_operators,
                evolution_problem.truncation_threshold,
            )

        return EvolutionResult(evolved_state, evaluated_aux_ops)

    @staticmethod
    def _summed_op_to_pauli_sum_op(
        hamiltonian: SummedOp,
    ) -> Union[PauliSumOp, PauliOp]:
        """
        Tries binding parameters in a Hamiltonian.

        Args:
            hamiltonian: The Hamiltonian that defines an evolution.

        Returns:
            Hamiltonian.

        Raises:
            ValueError: If the ``SummedOp`` Hamiltonian contains operators of an invalid type.
        """
        # PauliSumOp does not allow parametrized coefficients but after binding the parameters
        # we need to convert it into a PauliSumOp for the PauliEvolutionGate.
        op_list = []
        for op in hamiltonian.oplist:
            if not isinstance(op, PauliOp):
                raise ValueError(
                    "Content of the Hamiltonian not of type PauliOp. The "
                    f"following type detected: {type(op)}."
                )
            op_list.append(op)
        return sum(op_list)
