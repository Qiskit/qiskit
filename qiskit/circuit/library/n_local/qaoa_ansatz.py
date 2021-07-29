# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A generalized QAOA quantum circuit with a support of custom initial states and mixers."""

# pylint: disable=cyclic-import
from typing import Optional, List, Tuple
import numpy as np

from qiskit.circuit.library.evolved_operator_ansatz import EvolvedOperatorAnsatz, _is_pauli_identity
from qiskit.circuit.parametervector import ParameterVector
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister


class QAOAAnsatz(EvolvedOperatorAnsatz):
    """A generalized QAOA quantum circuit with a support of custom initial states and mixers.

    References:

        [1]: Farhi et al., A Quantum Approximate Optimization Algorithm.
            `arXiv:1411.4028 <https://arxiv.org/pdf/1411.4028>`_
    """

    def __init__(
        self,
        cost_operator=None,
        reps: int = 1,
        initial_state: Optional[QuantumCircuit] = None,
        mixer_operator=None,
        name: str = "QAOA",
    ):
        r"""
        Args:
            cost_operator (OperatorBase, optional): The operator representing the cost of
                the optimization problem, denoted as :math:`U(C, \gamma)` in the original paper.
                Must be set either in the constructor or via property setter.
            reps (int): The integer parameter p, which determines the depth of the circuit,
                as specified in the original paper, default is 1.
            initial_state (QuantumCircuit, optional): An optional initial state to use.
                If `None` is passed then a set of Hadamard gates is applied as an initial state
                to all qubits.
            mixer_operator (OperatorBase or QuantumCircuit, optional): An optional custom mixer
                to use instead of the global X-rotations, denoted as :math:`U(B, \beta)`
                in the original paper. Can be an operator or an optionally parameterized quantum
                circuit.
            name (str): A name of the circuit, default 'qaoa'
        """
        super().__init__(reps=reps, name=name)

        self._cost_operator = None
        self._reps = reps
        self._initial_state = initial_state
        self._mixer = mixer_operator

        # set this circuit as a not-built circuit
        # self._num_parameters = 0
        self._bounds = None

        # store cost operator and set the registers if the operator is not None
        self.cost_operator = cost_operator

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        valid = True

        super()._check_configuration()

        if self.cost_operator is None:
            valid = False
            if raise_on_failure:
                raise ValueError(
                    "The operator representing the cost of the optimization problem is not set"
                )

        if self.initial_state is not None and self.initial_state.num_qubits != self.num_qubits:
            valid = False
            if raise_on_failure:
                raise ValueError(
                    "The number of qubits of the initial state {} does not match "
                    "the number of qubits of the cost operator {}".format(
                        self.initial_state.num_qubits, self.num_qubits
                    )
                )

        if self.mixer_operator is not None and self.mixer_operator.num_qubits != self.num_qubits:
            valid = False
            if raise_on_failure:
                raise ValueError(
                    "The number of qubits of the mixer {} does not match "
                    "the number of qubits of the cost operator {}".format(
                        self.mixer_operator.num_qubits, self.num_qubits
                    )
                )

        return valid

    @property
    def parameter_bounds(self) -> List[Tuple[Optional[float], Optional[float]]]:
        """The parameter bounds for the unbound parameters in the circuit.

        Returns:
            A list of pairs indicating the bounds, as (lower, upper). None indicates an unbounded
            parameter in the corresponding direction. If None is returned, problem is fully
            unbounded.
        """
        if self._bounds is not None:
            return self._bounds

        # if the mixer is a circuit, we set no bounds
        if isinstance(self.mixer_operator, QuantumCircuit):
            return None

        # default bounds: None for gamma (cost operator), [0, 2pi] for gamma (mixer operator)
        beta_bounds = (0, 2 * np.pi)
        gamma_bounds = (None, None)
        bounds = []
        if not _is_pauli_identity(self.cost_operator):
            bounds += self.reps * [gamma_bounds]

        if not _is_pauli_identity(self.mixer_operator):
            bounds += self.reps * [beta_bounds]

        return bounds

    @parameter_bounds.setter
    def parameter_bounds(self, bounds: List[Tuple[Optional[float], Optional[float]]]) -> None:
        """Set the parameter bounds.

        Args:
            bounds: The new parameter bounds.
        """
        self._bounds = bounds

    @property
    def operators(self):
        """The operators that are evolved in this circuit.

        Returns:
             List[Union[OperatorBase, QuantumCircuit]]: The operators to be evolved (and circuits)
                in this ansatz.
        """
        return [self.cost_operator, self.mixer_operator]

    @property
    def cost_operator(self):
        """Returns an operator representing the cost of the optimization problem.

        Returns:
            OperatorBase: cost operator.
        """
        return self._cost_operator

    @cost_operator.setter
    def cost_operator(self, cost_operator) -> None:
        """Sets cost operator.

        Args:
            cost_operator (OperatorBase, optional): cost operator to set.
        """
        self._cost_operator = cost_operator
        self.qregs = [QuantumRegister(self.num_qubits, name="q")]
        self._invalidate()

    @property
    def reps(self) -> int:
        """Returns the `reps` parameter, which determines the depth of the circuit."""
        return self._reps

    @reps.setter
    def reps(self, reps: int) -> None:
        """Sets the `reps` parameter."""
        self._reps = reps
        self._invalidate()

    @property
    def initial_state(self) -> Optional[QuantumCircuit]:
        """Returns an optional initial state as a circuit"""
        if self._initial_state is not None:
            return self._initial_state

        # if no initial state is passed and we know the number of qubits, then initialize it.
        if self.num_qubits > 0:
            initial_state = QuantumCircuit(self.num_qubits)
            initial_state.h(range(self.num_qubits))
            return initial_state

        # otherwise we cannot provide a default
        return None

    @initial_state.setter
    def initial_state(self, initial_state: Optional[QuantumCircuit]) -> None:
        """Sets initial state."""
        self._initial_state = initial_state
        self._invalidate()

    # we can't directly specify OperatorBase as a return type, it causes a circular import
    # and pylint objects if return type is not documented
    @property
    def mixer_operator(self):
        """Returns an optional mixer operator expressed as an operator or a quantum circuit.

        Returns:
            OperatorBase or QuantumCircuit, optional: mixer operator or circuit.
        """
        if self._mixer is not None:
            return self._mixer

        # if no mixer is passed and we know the number of qubits, then initialize it.
        if self.cost_operator is not None:
            # local imports to avoid circular imports
            from qiskit.opflow import I, X

            num_qubits = self.cost_operator.num_qubits

            # Mixer is just a sum of single qubit X's on each qubit. Evolving by this operator
            # will simply produce rx's on each qubit.
            mixer_terms = [
                (I ^ left) ^ X ^ (I ^ (num_qubits - left - 1)) for left in range(num_qubits)
            ]
            mixer = sum(mixer_terms)
            return mixer

        # otherwise we cannot provide a default
        return None

    @mixer_operator.setter
    def mixer_operator(self, mixer_operator) -> None:
        """Sets mixer operator.

        Args:
            mixer_operator (OperatorBase or QuantumCircuit, optional): mixer operator or circuit
                to set.
        """
        self._mixer = mixer_operator
        self._invalidate()

    def _build(self):
        if self._data is not None:
            return

        super()._build()

        # keep old parameter order: first cost operator, then mixer operators
        num_cost = 0 if _is_pauli_identity(self.cost_operator) else 1
        if isinstance(self.mixer_operator, QuantumCircuit):
            num_mixer = self.mixer_operator.num_parameters
        else:
            num_mixer = 0 if _is_pauli_identity(self.mixer_operator) else 1

        num_parameters = self.reps * (num_cost + num_mixer)
        theta = ParameterVector("θ", num_parameters)

        # 0 3 1 4 2 5
        # 0 4 5 1 6 7 3
        # first the cost operator parameters then mixer operators
        reordered = [None] * num_parameters
        theta_iter = iter(theta)

        reordered = [next(theta_iter) for _ in range(self.reps * num_cost)]

        for i in range(self.reps):
            start_idx = i * num_mixer + (i + 1) * num_cost
            reordered[start_idx:start_idx] = [next(theta_iter) for _ in range(num_mixer)]

        self.assign_parameters(dict(zip(self.ordered_parameters, reordered)), inplace=True)
