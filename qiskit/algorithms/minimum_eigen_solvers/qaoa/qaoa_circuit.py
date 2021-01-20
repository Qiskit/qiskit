from typing import Optional, Union, cast, Set

import numpy as np
from qiskit.circuit import Parameter

from qiskit import QuantumCircuit
from qiskit.opflow import OperatorBase, I, X, CircuitStateFn, H, CircuitOp, EvolutionFactory

from qiskit.circuit.library import BlueprintCircuit


class QAOACircuit(BlueprintCircuit):
    """Global X phases and parameterized problem hamiltonian."""

    def __init__(self,
                 cost_operator: OperatorBase,
                 p: int,
                 initial_state: Optional[QuantumCircuit] = None,
                 mixer_operator: Optional[Union[QuantumCircuit, OperatorBase]] = None,
                 name: str = 'qaoa'):
        """
        Constructor, following the QAOA paper https://arxiv.org/abs/1411.4028

        Args:
            cost_operator: The operator representing the cost of
                            the optimization problem,
                            denoted as U(B, gamma) in the original paper.
            p: The integer parameter p, which determines the depth of the circuit,
                as specified in the original paper.
            initial_state: An optional initial state to use.
            mixer_operator: An optional custom mixer to use instead of the global X-rotations,
                            denoted as U(B, beta) in the original paper. Can be an operator or
                            an optionally parameterized quantum circuit.
        Raises:
            TypeError: invalid input
        """
        super().__init__(name=name)
        self._cost_operator = cost_operator
        self._num_qubits = cost_operator.num_qubits
        self._p = p
        self._initial_state = initial_state

        if isinstance(mixer_operator, QuantumCircuit):
            self._num_parameters = (1 + mixer_operator.num_parameters) * p
            self._bounds = [(None, None)] * p + [(None, None)] * p * mixer_operator.num_parameters
            self._mixer = mixer_operator
        elif isinstance(mixer_operator, OperatorBase):
            self._num_parameters = 2 * p
            self._bounds = [(None, None)] * p + [(None, None)] * p
            self._mixer = mixer_operator
        elif mixer_operator is None:
            self._num_parameters = 2 * p
            # next three lines are to avoid a mypy error (incorrect types, etc)
            self._bounds = []
            self._bounds.extend([(None, None)] * p)
            self._bounds.extend([(0, 2 * np.pi)] * p)
            # Mixer is just a sum of single qubit X's on each qubit. Evolving by this operator
            # will simply produce rx's on each qubit.
            num_qubits = self._cost_operator.num_qubits
            mixer_terms = [(I ^ left) ^ X ^ (I ^ (num_qubits - left - 1))
                           for left in range(num_qubits)]
            self._mixer = sum(mixer_terms)

        self.support_parameterized_circuit = True

        # todo: this is from blueprint
        self._data = None

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        pass

    def _build(self) -> None:
        """Build the circuit."""
        if self._data:
            return

        _ = self._check_configuration()

        self._data = []

    @property
    def parameters(self) -> Set[Parameter]:
        """Get the :class:`~qiskit.circuit.Parameter` objects in the circuit.

        Returns:
            A set containing the unbound circuit parameters.
        """
        self._build()
        return super().parameters

    # todo: q=None is not required
    def construct_circuit(self, parameters, q=None):
        """ construct circuit """
        # todo: self._num_parameters with a call to circuit
        if not len(parameters) == self._num_parameters:
            raise ValueError('Incorrect number of angles: expecting {}, but {} given.'.format(
                self._num_parameters, len(parameters)
            ))

        # initialize circuit, possibly based on given register/initial state
        if isinstance(self._initial_state, QuantumCircuit):
            circuit_op = CircuitStateFn(self._initial_state)
        elif self._initial_state is not None:   # todo: this condition should be always false
            circuit_op = CircuitStateFn(self._initial_state.construct_circuit('circuit'))
        else:
            circuit_op = (H ^ self._num_qubits)

        # iterate over layers
        for idx in range(self._p):
            # the first [:self._p] parameters are used for the cost operator,
            # so we apply them here
            circuit_op = (self._cost_operator * parameters[idx]).exp_i().compose(circuit_op)
            if isinstance(self._mixer, OperatorBase):
                mixer = cast(OperatorBase, self._mixer)
                # we apply beta parameter in case of operator based mixer.
                circuit_op = (mixer * parameters[idx + self._p]).exp_i().compose(circuit_op)
            else:
                # mixer as a quantum circuit that can be parameterized
                mixer = cast(QuantumCircuit, self._mixer)
                num_params = mixer.num_parameters
                # the remaining [self._p:] parameters are used for the mixer,
                # there may be multiple layers, so parameters are grouped by layers.
                param_values = parameters[self._p + num_params * idx:
                                          self._p + num_params * (idx + 1)]
                param_dict = dict(zip(mixer.parameters, param_values))
                mixer = mixer.assign_parameters(param_dict)
                circuit_op = CircuitOp(mixer).compose(circuit_op)

        evolution = EvolutionFactory.build(self._cost_operator)
        circuit_op = evolution.convert(circuit_op)
        return circuit_op.to_circuit()
