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
from qiskit.circuit.library.evolved_operator_ansatz import EvolvedOperatorGate
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
        self._bounds = None

        # store cost operator and set the registers if the operator is not None
        self.cost_operator = cost_operator

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        valid = True

        if not super()._check_configuration(raise_on_failure):
            return False

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
    def parameter_bounds(self) -> Optional[List[Tuple[Optional[float], Optional[float]]]]:
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

        if not _is_pauli_identity(self.mixer_operator):
            bounds += self.reps * [beta_bounds]

        if not _is_pauli_identity(self.cost_operator):
            bounds += self.reps * [gamma_bounds]

        return bounds

    @parameter_bounds.setter
    def parameter_bounds(
        self, bounds: Optional[List[Tuple[Optional[float], Optional[float]]]]
    ) -> None:
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
        if self.cost_operator is not None:
            return _get_default_initial_state(self.cost_operator)

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
            return _get_default_mixer(self.cost_operator)

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

    def _build_gate(self):
        """Return a QAOAGate based on the current settings."""
        return QAOAGate(
            cost_operator=self.cost_operator,
            reps=self.reps,
            initial_state=self.initial_state,
            mixer_operator=self.mixer_operator,
            label=self.name,
        )

    @property
    def num_qubits(self) -> int:
        if self._cost_operator is None:
            return 0
        return self._cost_operator.num_qubits

    def _build(self):
        super()._build()

        num_cost = 0 if _is_pauli_identity(self.cost_operator) else 1
        if isinstance(self.mixer_operator, QuantumCircuit):
            num_mixer = self.mixer_operator.num_parameters
        else:
            num_mixer = 0 if _is_pauli_identity(self.mixer_operator) else 1

        betas = ParameterVector("β", self.reps * num_mixer)
        gammas = ParameterVector("γ", self.reps * num_cost)

        # Create a permutation to take us from (cost_1, mixer_1, cost_2, mixer_2, ...)
        # to (cost_1, cost_2, ..., mixer_1, mixer_2, ...), or if the mixer is a circuit
        # with more than 1 parameters, from (cost_1, mixer_1a, mixer_1b, cost_2, ...)
        # to (cost_1, cost_2, ..., mixer_1a, mixer_1b, mixer_2a, mixer_2b, ...)
        reordered = []
        for rep in range(self.reps):
            reordered.extend(gammas[rep * num_cost : (rep + 1) * num_cost])
            reordered.extend(betas[rep * num_mixer : (rep + 1) * num_mixer])

        self.assign_parameters(reordered, inplace=True)


class QAOAGate(EvolvedOperatorGate):
    """A generalized QAOA gate with a support of custom initial states and mixers.

    References:

        [1]: Farhi et al., A Quantum Approximate Optimization Algorithm.
            `arXiv:1411.4028 <https://arxiv.org/pdf/1411.4028>`_
    """

    def __init__(
        self,
        cost_operator,
        reps: int = 1,
        initial_state: Optional[QuantumCircuit] = None,
        mixer_operator=None,
        label: str = None,
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
            label (str): A label.
        Raises:
            AttributeError: when cost_operator and initial_state are not compatible
        """
        self.cost_operator = cost_operator
        self.reps = reps

        if initial_state is None:
            initial_state = _get_default_initial_state(cost_operator)
        else:
            if initial_state.num_qubits != cost_operator.num_qubits:
                raise AttributeError(
                    "initial_state and cost_operator has incompatible number of qubits"
                )

        self.initial_state = initial_state

        if mixer_operator is None:
            mixer_operator = _get_default_mixer(cost_operator)

        self.mixer_operator = mixer_operator

        super().__init__(
            operators=[cost_operator, mixer_operator],
            reps=reps,
            initial_state=initial_state,
            label=label,
        )


def _get_default_mixer(cost_operator):
    # local imports to avoid circular imports
    from qiskit.opflow import I, X

    num_qubits = cost_operator.num_qubits

    # Mixer is just a sum of single qubit X's on each qubit. Evolving by this operator
    # will simply produce rx's on each qubit.
    active_indices = _active_qubits(cost_operator)

    if len(active_indices) == 0:
        return 0 * (I ^ num_qubits)

    mixer_terms = [(I ^ left) ^ X ^ (I ^ (num_qubits - left - 1)) for left in active_indices]
    return sum(mixer_terms)


def _get_default_initial_state(cost_operator):
    initial_state = QuantumCircuit(cost_operator.num_qubits)
    active_indices = _active_qubits(cost_operator)

    if len(active_indices) > 0:
        # Opflow indices are reversed with respect to circuit indices
        active_indices = [cost_operator.num_qubits - 1 - index for index in active_indices]
        initial_state.h(active_indices)

    return initial_state


def _active_qubits(operator):
    from qiskit.opflow import PauliSumOp, PauliOp

    # active qubit selection only supported for PauliSumOps
    if isinstance(operator, PauliSumOp):
        sparse_pauli = operator.primitive
        paulis = sparse_pauli.paulis.to_labels()
    elif isinstance(operator, PauliOp):
        paulis = [operator.primitive.to_label()]
    else:
        return list(range(operator.num_qubits))

    # for each Pauli string get a list which Pauli is the identity (i.e. not active)
    is_identity = [list(map(lambda pauli: pauli == "I", pauli_string)) for pauli_string in paulis]

    # use numpy act a logical and on each index across the Pauli strings
    idle_qubits = np.all(np.array(is_identity), axis=0)

    # return the indices of the qubits that are not idle
    active_indices = [index for index, idle in enumerate(idle_qubits) if not idle]

    return active_indices
