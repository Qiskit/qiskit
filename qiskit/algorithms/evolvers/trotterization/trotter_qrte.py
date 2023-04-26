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

from __future__ import annotations

import warnings

from qiskit import QuantumCircuit
from qiskit.algorithms.aux_ops_evaluator import eval_observables
from qiskit.algorithms.evolvers import EvolutionProblem, EvolutionResult
from qiskit.algorithms.evolvers.real_evolver import RealEvolver
from qiskit.opflow import (
    SummedOp,
    PauliOp,
    CircuitOp,
    ExpectationBase,
    CircuitSampler,
    PauliSumOp,
    StateFn,
    OperatorBase,
)
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.providers import Backend
from qiskit.synthesis import ProductFormula, LieTrotter
from qiskit.utils import QuantumInstance
from qiskit.utils.deprecation import deprecate_func


class TrotterQRTE(RealEvolver):
    """Deprecated: Quantum Real Time Evolution using Trotterization.

    The TrotterQRTE class has been superseded by the
    :class:`qiskit.algorithms.time_evolvers.trotterization.TrotterQRTE` class.
    This class will be deprecated in a future release and subsequently
    removed after that.

    Type of Trotterization is defined by a ProductFormula provided.

    Examples::

            from qiskit.opflow import X, Z, Zero
            from qiskit.algorithms import EvolutionProblem, TrotterQRTE
            from qiskit import BasicAer
            from qiskit.utils import QuantumInstance

            operator = X + Z
            initial_state = Zero
            time = 1
            evolution_problem = EvolutionProblem(operator, 1, initial_state)
            # LieTrotter with 1 rep
            backend = BasicAer.get_backend("statevector_simulator")
            quantum_instance = QuantumInstance(backend=backend)
            trotter_qrte = TrotterQRTE(quantum_instance=quantum_instance)
            evolved_state = trotter_qrte.evolve(evolution_problem).evolved_state
    """

    @deprecate_func(
        additional_msg=(
            "Instead, use the class ``qiskit.algorithms.time_evolvers.trotterization.TrotterQRTE``."
            " See https://qisk.it/algo_migration for a migration guide."
        ),
        since="0.24.0",
    )
    def __init__(
        self,
        product_formula: ProductFormula | None = None,
        expectation: ExpectationBase | None = None,
        quantum_instance: QuantumInstance | Backend | None = None,
    ) -> None:
        """
        Args:
            product_formula: A Lie-Trotter-Suzuki product formula. The default is the Lie-Trotter
                first order product formula with a single repetition.
            expectation: An instance of ExpectationBase which defines a method for calculating
                expectation values of EvolutionProblem.aux_operators.
            quantum_instance: A quantum instance used for calculating expectation values of
                EvolutionProblem.aux_operators.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            super().__init__()
        if product_formula is None:
            product_formula = LieTrotter()
        self._product_formula = product_formula
        self._quantum_instance = None
        self._circuit_sampler: CircuitSampler | None = None
        if quantum_instance is not None:
            self.quantum_instance = quantum_instance
        self._expectation = expectation

    @property
    def product_formula(self) -> ProductFormula:
        """Returns a product formula used in the algorithm."""
        return self._product_formula

    @product_formula.setter
    def product_formula(self, product_formula: ProductFormula) -> None:
        """
        Sets a product formula.
        Args:
            product_formula: A formula that defines the Trotterization algorithm.
        """
        self._product_formula = product_formula

    @property
    def quantum_instance(self) -> QuantumInstance | None:
        """Returns a quantum instance used in the algorithm."""
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, quantum_instance: QuantumInstance | Backend | None) -> None:
        """
        Sets a quantum instance and a circuit sampler.
        Args:
            quantum_instance: The quantum instance used to run this algorithm.
        """
        if isinstance(quantum_instance, Backend):
            quantum_instance = QuantumInstance(quantum_instance)

        self._circuit_sampler = None
        if quantum_instance is not None:
            self._circuit_sampler = CircuitSampler(quantum_instance)

        self._quantum_instance = quantum_instance

    @property
    def expectation(self) -> ExpectationBase | None:
        """Returns an expectation used in the algorithm."""
        return self._expectation

    @expectation.setter
    def expectation(self, expectation: ExpectationBase | None) -> None:
        """
        Sets an expectation.
        Args:
            expectation: An instance of ExpectationBase which defines a method for calculating
                expectation values of EvolutionProblem.aux_operators.
        """
        self._expectation = expectation

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
        evaluated on an evolved state using a backend provided.

        .. note::
            Time-dependent Hamiltonians are not yet supported.

        Args:
            evolution_problem: Instance defining evolution problem. For the included Hamiltonian,
                ``PauliOp``, ``SummedOp`` or ``PauliSumOp`` are supported by TrotterQRTE.

        Returns:
            Evolution result that includes an evolved state as a quantum circuit and, optionally,
            auxiliary operators evaluated for a resulting state on a backend.

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

        if evolution_problem.aux_operators is not None and (
            self._quantum_instance is None or self._expectation is None
        ):
            raise ValueError(
                "aux_operators were provided for evaluations but no ``expectation`` or "
                "``quantum_instance`` was provided."
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
            PauliEvolutionGate(hamiltonian, evolution_problem.time, synthesis=self._product_formula)
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
                self._quantum_instance,
                evolved_state.primitive,
                evolution_problem.aux_operators,
                self._expectation,
                evolution_problem.truncation_threshold,
            )

        return EvolutionResult(evolved_state, evaluated_aux_ops)

    @staticmethod
    def _summed_op_to_pauli_sum_op(
        hamiltonian: SummedOp,
    ) -> PauliSumOp | PauliOp:
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
