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

from qiskit.circuit import ParameterVector, Parameter
from qiskit.circuit.library import PauliEvolutionGate, MatrixEvolutionGate
from qiskit.quantum_info import Operator

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.gate import Gate

from qiskit.circuit.library.blueprintcircuit import BlueprintCircuit


class QAOAAnsatz(BlueprintCircuit):
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
        super().__init__(name=name)

        self._qaoa = QAOAGate(
            cost_operator=cost_operator,
            reps=reps,
            initial_state=initial_state,
            mixer_operator=mixer_operator,
            label=name
        )

        self.cost_operator = cost_operator

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        try:
            self._qaoa._check_configuration()
        except ValueError as exc:
            if raise_on_failure:
                raise exc
            return False

        return True

    @property
    def operators(self):
        """The operators that are evolved in this circuit.

        Returns:
             List[Union[OperatorBase, QuantumCircuit]]: The operators to be evolved (and circuits)
                in this ansatz.
        """
        return [self.cost_operator, self.mixer_operator]

    @property
    def parameter_bounds(self) -> Optional[List[Tuple[Optional[float], Optional[float]]]]:
        """The parameter bounds for the unbound parameters in the circuit.

        Returns:
            A list of pairs indicating the bounds, as (lower, upper). None indicates an unbounded
            parameter in the corresponding direction. If None is returned, problem is fully
            unbounded.
        """
        return self._qaoa.parameter_bounds

    @parameter_bounds.setter
    def parameter_bounds(
        self, bounds: Optional[List[Tuple[Optional[float], Optional[float]]]]
    ) -> None:
        """Set the parameter bounds.

        Args:
            bounds: The new parameter bounds.
        """
        self._qaoa.parameter_bounds = bounds

    @property
    def cost_operator(self):
        """Returns an operator representing the cost of the optimization problem.

        Returns:
            OperatorBase: cost operator.
        """
        return self._qaoa.cost_operator

    @cost_operator.setter
    def cost_operator(self, cost_operator) -> None:
        """Sets cost operator.

        Args:
            cost_operator (OperatorBase, optional): cost operator to set.
        """
        self._qaoa.cost_operator = cost_operator
        self._cost_operator = cost_operator
        self.qregs = [QuantumRegister(self.num_qubits, name="q")]
        self._invalidate()

    @property
    def reps(self) -> int:
        """Returns the `reps` parameter, which determines the depth of the circuit."""
        return self._qaoa.reps

    @reps.setter
    def reps(self, reps: int) -> None:
        """Sets the `reps` parameter."""
        self._qaoa.reps = reps
        self._invalidate()

    @property
    def initial_state(self) -> Optional[QuantumCircuit]:
        """Returns an optional initial state as a circuit"""
        return self._qaoa.initial_state

    @initial_state.setter
    def initial_state(self, initial_state: Optional[QuantumCircuit]) -> None:
        """Sets initial state."""
        self._qaoa.initial_state = initial_state
        self._invalidate()

    # we can't directly specify OperatorBase as a return type, it causes a circular import
    # and pylint objects if return type is not documented
    @property
    def mixer_operator(self):
        """Returns an optional mixer operator expressed as an operator or a quantum circuit.

        Returns:
            OperatorBase or QuantumCircuit, optional: mixer operator or circuit.
        """
        return self._qaoa.mixer_operator

    @mixer_operator.setter
    def mixer_operator(self, mixer_operator) -> None:
        """Sets mixer operator.

        Args:
            mixer_operator (OperatorBase or QuantumCircuit, optional): mixer operator or circuit
                to set.
        """
        self._qaoa.mixer = mixer_operator
        self._invalidate()

    @property
    def num_qubits(self) -> int:
        return self._qaoa.num_qubits

    def _build(self):
        if self._data is not None:
            return

        self._data = []
        self.append(self._qaoa, self.qubits)


class QAOAGate(Gate):
    """A gate implementing the QAOA ansatz."""

    def __init__(
        self,
        cost_operator=None,
        reps: int = 1,
        initial_state: Optional[QuantumCircuit] = None,
        mixer_operator=None,
        label: str = "QAOA",
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
        num_qubits = 0 if cost_operator is None else cost_operator.num_qubits
        super().__init__("QAOAGate", num_qubits, [], label=label)

        self._initial_state = initial_state
        self._bounds = None

        self._reps = None
        self.reps = reps

        self._cost_operator = cost_operator
        self._mixer_operator = mixer_operator
        self._update_parameters()

    @property
    def cost_operator(self):
        """Return the cost operator."""
        return self._cost_operator

    @cost_operator.setter
    def cost_operator(self, cost_operator):
        """Set the cost operator."""
        if cost_operator != self._cost_operator:
            self.definition = None
            self.num_qubits = cost_operator.num_qubits
            self._cost_operator = cost_operator
            self._update_parameters()

    @property
    def mixer_operator(self):
        """Return the mixer operator. Provides a default if none is set."""
        if self._mixer_operator is not None:
            return self._mixer_operator

        if self._cost_operator is not None:
            from qiskit.opflow import PauliSumOp
            paulis = [("".join(["I"] * i + ["X"] + ["I"] * (self.num_qubits - 1 - i)), 1)
                      for i in range(self.num_qubits)]
            return PauliSumOp.from_list(paulis)

        return None

    @mixer_operator.setter
    def mixer_operator(self, mixer_operator):
        """Set the mixer operator."""
        if mixer_operator != self._mixer_operator:
            self.definition = None
            self._mixer_operator = mixer_operator
            self._update_parameters()

    @property
    def reps(self):
        """Return the number of repetitions."""
        return self._reps

    @reps.setter
    def reps(self, reps):
        """Set the number of repetitions."""
        if reps < 0:
            raise ValueError(f"Number of repetitions must be positive, but is {reps}.")

        if reps != self._reps:
            self.definition = None
            self._reps = reps

    @property
    def initial_state(self):
        """Return the initial state. Provides a default if none is set."""
        if self._initial_state is not None:
            return self._initial_state

        if self.num_qubits == 0:
            return QuantumCircuit()

        initial_state = QuantumCircuit(self.num_qubits)
        initial_state.h(initial_state.qubits)
        return initial_state

    @initial_state.setter
    def initial_state(self, initial_state):
        """Set the initial state."""
        if initial_state != self._initial_state:
            self.defintion = None
            self._initial_state = initial_state

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

        bounds = self.reps * [beta_bounds]
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

    def _update_parameters(self):
        if isinstance(self.mixer_operator, QuantumCircuit):
            num_mixer = self.mixer_operator.num_parameters
        else:
            num_mixer = 1

        self.betas = ParameterVector("β", self.reps * num_mixer)
        self.gammas = ParameterVector("γ", self.reps)

        # Create the list of parameters as (cost_1, mixer_1, cost_2, mixer_2, ...)
        # or, with more than one parameter as as (cost_1, mixer_1a, mixer_1b, cost_2, ...)
        self.params = []
        for rep in range(self.reps):
            self.params.extend(self.gammas[rep : (rep + 1)])
            self.params.extend(self.betas[rep * num_mixer : (rep + 1) * num_mixer])

    def _check_configuration(self):
        if self.cost_operator is None:
            raise ValueError(
                "The operator representing the cost of the optimization problem is not set"
            )

        if self.initial_state.num_qubits != self.num_qubits:
            raise ValueError(
                f"The number of qubits of the initial state {self.initial_state.num_qubits} does "
                f"not match the number of qubits of the cost operator {self.num_qubits}."
            )

        if self.mixer_operator.num_qubits != self.num_qubits:
            raise ValueError(
                f"The number of qubits of the mixer {self.mixer_operator.num_qubits} does not "
                f"match the number of qubits of the cost operator {self.num_qubits}."
            )

    def _define(self):
        self._check_configuration()
        definition = QuantumCircuit(self.num_qubits)

        if isinstance(self.mixer_operator, QuantumCircuit):
            num_mixer = self.mixer_operator.num_qubits
        else:
            num_mixer = 1

        # add initial state
        definition.compose(self.initial_state, inplace=True)

        # add evolutions
        for rep in range(self.reps):
            definition.compose(
                _get_evolved_operator(self.cost_operator, self.gammas[rep : (rep + 1)]),
                inplace=True
            )
            definition.compose(
                _get_evolved_operator(self.mixer_operator,
                                      self.betas[rep * num_mixer : (rep + 1) * num_mixer]),
                inplace=True
            )

        self.definition = definition


def _get_evolved_operator(operator, parameter):
    if isinstance(operator, QuantumCircuit):
        return operator.assign_parameters(parameter)

    if isinstance(operator, (Operator, np.ndarray)):
        inst = MatrixEvolutionGate(operator, time=parameter)
    else:
        inst = PauliEvolutionGate(operator, time=parameter)

    evolved = QuantumCircuit(inst.num_qubits)
    evolved.append(inst, evolved.qubits)
    return evolved
