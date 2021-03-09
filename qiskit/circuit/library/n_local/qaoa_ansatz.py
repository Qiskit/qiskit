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
from typing import Optional, cast, Set, List, Tuple

import numpy as np

from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parametervector import ParameterVector
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from ..blueprintcircuit import BlueprintCircuit


class QAOAAnsatz(BlueprintCircuit):
    """A generalized QAOA quantum circuit with a support of custom initial states and mixers.

    References:

        [1]: Farhi et al., A Quantum Approximate Optimization Algorithm.
            `arXiv:1411.4028 <https://arxiv.org/pdf/1411.4028>`_
    """

    def __init__(self,
                 cost_operator=None,
                 reps: int = 1,
                 initial_state: Optional[QuantumCircuit] = None,
                 mixer_operator=None,
                 name: str = "qaoa"):
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

        self._cost_operator = None
        self._reps = reps
        self._initial_state = initial_state
        self._mixer = mixer_operator

        # set this circuit as a not-built circuit
        self._num_parameters = 0
        self._bounds = None

        # store cost operator and set the registers if the operator is not None
        self.cost_operator = cost_operator

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        valid = True

        if self._cost_operator is None:
            valid = False
            if raise_on_failure:
                raise AttributeError("The operator representing the cost of "
                                     "the optimization problem is not set")
        if self._reps is None or self._reps < 1:
            valid = False
            if raise_on_failure:
                raise AttributeError("The integer parameter reps, which determines the depth "
                                     "of the circuit, needs to be >= 1 but has value {}"
                                     .format(self._reps))

        if self.initial_state is not None and self.initial_state.num_qubits != self.num_qubits:
            valid = False
            if raise_on_failure:
                raise AttributeError("The number of qubits of the initial state {} does not match "
                                     "the number of qubits of the cost operator {}"
                                     .format(self.initial_state.num_qubits, self.num_qubits))

        if self.mixer_operator is not None and self.mixer_operator.num_qubits != self.num_qubits:
            valid = False
            if raise_on_failure:
                raise AttributeError("The number of qubits of the mixer {} does not match "
                                     "the number of qubits of the cost operator {}"
                                     .format(self.mixer_operator.num_qubits, self.num_qubits))

        return valid

    def _build(self) -> None:
        """Build the circuit."""
        if self._data:
            return

        self._check_configuration()
        self._data = []

        # calculate bounds, num_parameters, mixer
        self._calculate_parameters()

        # parameterize circuit and build it
        param_vector = ParameterVector("θ", self._num_parameters)
        circuit = self._construct_circuit(param_vector)

        # append(replace) the circuit to this
        self.compose(circuit, inplace=True)

    def _reset_registers(self, num_qubits):
        """Set the registers and qubits to the new size."""
        self._qregs = []
        self._qubits = []
        self._qubit_set = set()

        if num_qubits > 0:
            qr = QuantumRegister(num_qubits, 'q')
            self.add_register(qr)

    @property
    def parameters(self) -> Set[Parameter]:
        """Get the :class:`~qiskit.circuit.Parameter` objects in the circuit.

        Returns:
            A set containing the unbound circuit parameters.
        """
        self._build()
        return super().parameters

    @property
    def parameter_bounds(self) -> List[Tuple[float, float]]:
        """Parameter bounds.

        Returns: A list of pairs indicating the bounds, as (lower, upper). None indicates
            an unbounded parameter in the corresponding direction. If None is returned, problem is
            fully unbounded or is not built yet.
        """

        return self._bounds

    def _calculate_parameters(self):
        """Calculated internal parameters of the circuit to be built."""
        from qiskit.opflow import OperatorBase
        if isinstance(self._mixer, QuantumCircuit):
            self._num_parameters = (1 + self._mixer.num_parameters) * self._reps
            self._bounds = [(None, None)] * self._reps + \
                           [(None, None)] * self._reps * self._mixer.num_parameters
        elif isinstance(self._mixer, OperatorBase):
            self._num_parameters = 2 * self._reps
            self._bounds = [(None, None)] * self._reps + [(None, None)] * self._reps
        elif self._mixer is None:
            self._num_parameters = 2 * self._reps
            self._bounds = [(None, None)] * self._reps + [(0, 2 * np.pi)] * self._reps

    def _construct_circuit(self, parameters) -> QuantumCircuit:
        """Construct a parameterized circuit."""
        if not len(parameters) == self._num_parameters:
            raise ValueError('Incorrect number of angles: expecting {}, but {} given.'.format(
                self._num_parameters, len(parameters)
            ))

        # local imports to avoid circular imports
        from qiskit.opflow import CircuitStateFn
        from qiskit.opflow import CircuitOp, EvolutionFactory
        from qiskit.opflow import OperatorBase

        circuit_op = CircuitStateFn(self.initial_state)

        # iterate over layers
        for idx in range(self._reps):
            # the first [:self._reps] parameters are used for the cost operator,
            # so we apply them here
            circuit_op = (self._cost_operator * parameters[idx]).exp_i().compose(circuit_op)
            mixer = self.mixer_operator
            if isinstance(mixer, OperatorBase):
                mixer = cast(OperatorBase, mixer)
                # we apply beta parameter in case of operator based mixer.
                circuit_op = (mixer * parameters[idx + self._reps]).exp_i().compose(circuit_op)
            else:
                # mixer as a quantum circuit that can be parameterized
                mixer = cast(QuantumCircuit, mixer)
                num_params = mixer.num_parameters
                # the remaining [self._p:] parameters are used for the mixer,
                # there may be multiple layers, so parameters are grouped by layers.
                param_values = parameters[self._reps + num_params * idx:
                                          self._reps + num_params * (idx + 1)]
                param_dict = dict(zip(mixer.parameters, param_values))
                mixer = mixer.assign_parameters(param_dict)
                circuit_op = CircuitOp(mixer).compose(circuit_op)

        evolution = EvolutionFactory.build(self._cost_operator)
        circuit_op = evolution.convert(circuit_op)
        return circuit_op.to_circuit()

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
        self._invalidate()

        num_qubits = cost_operator.num_qubits if cost_operator else None
        self._reset_registers(num_qubits)

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
    # pylint: disable=missing-return-type-doc
    @property
    def mixer_operator(self):
        """Returns an optional mixer operator expressed as an operator or a quantum circuit.

        Returns:
            OperatorBase or QuantumCircuit, optional: mixer operator or circuit.
        """
        if self._mixer is not None:
            return self._mixer

        # if no mixer is passed and we know the number of qubits, then initialize it.
        if self.num_qubits > 0:
            # local imports to avoid circular imports
            from qiskit.opflow import I, X
            # Mixer is just a sum of single qubit X's on each qubit. Evolving by this operator
            # will simply produce rx's on each qubit.
            mixer_terms = [(I ^ left) ^ X ^ (I ^ (self.num_qubits - left - 1))
                           for left in range(self.num_qubits)]
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
