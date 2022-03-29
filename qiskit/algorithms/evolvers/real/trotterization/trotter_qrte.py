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

"""An algorithm to implement a Trotterization real time-evolution."""

from typing import Union, Optional

from qiskit import QuantumCircuit
from qiskit.algorithms.aux_ops_evaluator import eval_observables
from qiskit.algorithms.evolvers import EvolutionProblem, EvolutionResult
from qiskit.algorithms.evolvers.real.real_evolver import RealEvolver
from qiskit.opflow import (
    SummedOp,
    PauliOp,
    CircuitOp,
    ExpectationBase,
    CircuitSampler,
    PauliSumOp,
    StateFn,
    ListOp,
    CircuitStateFn,
    OperatorBase,
)
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.providers import Backend, BaseBackend
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.synthesis import ProductFormula, LieTrotter
from qiskit.utils import QuantumInstance


class TrotterQRTE(RealEvolver):
    """Quantum Real Time Evolution using Trotterization.
    Type of Trotterization is defined by a ProductFormula provided.

    Examples:

        .. jupyter-execute::

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

    def __init__(
        self,
        product_formula: Optional[ProductFormula] = None,
        expectation: Optional[ExpectationBase] = None,
        quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
    ) -> None:
        """
        Args:
            product_formula: A Lie-Trotter-Suzuki product formula. The default is the Lie-Trotter
                first order product formula with a single repetition.
            expectation: An instance of ExpectationBase which defines a method for calculating
                expectation values of EvolutionProblem.aux_operators.
            quantum_instance: A quantum instance used for calculations. If not provided,
                calculations are performed classically which work reasonably only for small systems.
                In case of auxiliary operators provided in ``EvolutionProblem``, a quantum instance
                is required.

            .. note::

                Shot-based simulators like, e.g., the ``qasm_simulator`` will return counts sampled
                from an evolved state, not the description of the state itself as it happens for the
                ``statevector_simulator`` for example.
        """
        if product_formula is None:
            product_formula = LieTrotter()
        self._product_formula = product_formula
        self._quantum_instance = None
        self._circuit_sampler = None
        if quantum_instance is not None:
            self.quantum_instance = quantum_instance
        self._expectation = expectation

    @property
    def product_formula(self) -> ProductFormula:
        """Returns a product formula used in the algorithm."""
        return self._product_formula

    @property
    def quantum_instance(self) -> Union[QuantumInstance, BaseBackend, Backend]:
        """Returns a quantum instance used in the algorithm."""
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(
        self, quantum_instance: Union[QuantumInstance, BaseBackend, Backend]
    ) -> None:
        """Sets a quantum instance and a circuit sampler.
        Args:
            quantum_instance: The quantum instance used to run this algorithm.
        """
        if isinstance(quantum_instance, (BaseBackend, Backend)):
            quantum_instance = QuantumInstance(quantum_instance)

        self._circuit_sampler = None
        if quantum_instance is not None:
            self._circuit_sampler = CircuitSampler(quantum_instance)

        self._quantum_instance = quantum_instance

    @property
    def expectation(self) -> ExpectationBase:
        """Returns an expectation used in the algorithm."""
        return self._expectation

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
        based on a product formula provided.

        .. note::
            Time-dependent Hamiltonians are not yet supported.

        Args:
            evolution_problem: Instance defining evolution problem. For the included Hamiltonian,
                ``PauliOp``, ``SparsePauliOp``, ``Pauli`` or ``SummedOp`` thereof or ``PauliSumOp``
                are supported by TrotterQRTE.

        Returns:
            Evolution result that includes an evolved state.

        Raises:
            ValueError: If ``t_param`` is not set to None in the EvolutionProblem (feature not
                currently supported).
            ValueError: If the ``initial_state`` is not provided in the EvolutionProblem.
        """
        if evolution_problem.t_param is not None:
            raise ValueError(
                "TrotterQRTE does not accept a time dependent hamiltonian,"
                "``t_param`` from the EvolutionProblem should be set to None."
            )

        if evolution_problem.aux_operators is not None and (
            self._quantum_instance is None or self._expectation is None
        ):
            raise ValueError(
                "aux_operators where provided for evaluations but no ``expectation`` or "
                "``quantum_instance`` was provided."
            )
        hamiltonian = evolution_problem.hamiltonian
        if not isinstance(hamiltonian, (Pauli, PauliOp, SparsePauliOp, PauliSumOp, SummedOp)):
            raise ValueError(
                f"TrotterQRTE only accepts Pauli | PauliOp | SparsePauliOp | "
                f"PauliSumOp | SummedOp, {type(hamiltonian)} provided."
            )
        if isinstance(hamiltonian, OperatorBase):
            hamiltonian = hamiltonian.bind_parameters(evolution_problem.hamiltonian_value_dict)
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
            quantum_state = evolution_gate @ initial_state
            if self._circuit_sampler is not None:
                quantum_state = self._circuit_sampler.convert(quantum_state)

            evolved_state = quantum_state.eval()

        else:
            raise ValueError("``initial_state`` must be provided in the EvolutionProblem.")

        evaluated_aux_ops = None
        if evolution_problem.aux_operators is not None:
            evaluated_aux_ops = eval_observables(
                self._quantum_instance,
                quantum_state.primitive,
                evolution_problem.aux_operators,
                self._expectation,
                evolution_problem.truncation_threshold,
            )

        return EvolutionResult(evolved_state, evaluated_aux_ops)

    @staticmethod
    def _summed_op_to_pauli_sum_op(
        hamiltonian: Union[SummedOp],
    ) -> Union[PauliSumOp, PauliOp]:
        """
        Tries binding parameters in a Hamiltonian.

        Args:
            hamiltonian: The Hamiltonian of that defines an evolution.

        Returns:
            Hamiltonian.

        Raises:
            ValueError: If the ``SummedOp`` Hamiltonian contains operators of an invalid type.
        """
        # PauliSumOp does not allow parametrized coefficients but after binding the parameters
        # we need to convert it into a PauliSumOp for the PauliEvolutionGate.
        op_list = []
        for op in hamiltonian.oplist:
            if not isinstance(op, (PauliOp, Pauli, SparsePauliOp)):
                raise ValueError(
                    f"Content of the Hamiltonian not of type PauliOp, Pauli, or SparsePauliOp. The "
                    f"following type detected: {type(op)}."
                )
            op_list.append(op)
        return sum(op_list)
