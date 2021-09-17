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
from qiskit.opflow.primitive_ops.pauli_op import PauliOp
from typing import Optional, Set, List, Tuple

from qiskit.circuit.library.evolved_operator_ansatz import EvolvedOperatorAnsatz
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.exceptions import CircuitError
from qiskit.exceptions import QiskitError
import itertools
import numpy as np

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
            mixer_operator (OperatorBase or QuantumCircuit, List[QuantumCircuit], optional): An optional 
                custom mixer or list of mixer to be used instead of the global X-rotations, denoted 
                as :math:`U(B, \beta)` in the original paper. Can be an operator or an optionally 
                parameterized quantum circuit.
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

        if self.cost_operator is None:
            valid = False
            if raise_on_failure:
                raise AttributeError(
                    "The operator representing the cost of the optimization problem is not set"
                )

        if self.reps is None or self.reps < 0:
            valid = False
            if raise_on_failure:
                raise AttributeError(
                    "The integer parameter reps, which determines the depth "
                    "of the circuit, needs to be >= 0 but has value {}".format(self._reps)
                )

        if self.initial_state is not None and self.initial_state.num_qubits != self.num_qubits:
            valid = False
            if raise_on_failure:
                raise AttributeError(
                    "The number of qubits of the initial state {} does not match "
                    "the number of qubits of the cost operator {}".format(
                        self.initial_state.num_qubits, self.num_qubits
                    )
                )

        if self.mixer_operator is not None:
            if isinstance(self.mixer_operator,list):
                mixer_qubit_check = np.argwhere([_.num_qubits!=self.num_qubits for _ in self.mixer_operator] is True)
                if 0<len(mixer_qubit_check):
                    valid =  False
                    if raise_on_failure:
                        raise AttributeError("The number of qubits of the mixer operator(s) {} at "
                                            "argument(s) {} does not match the number of qubits of "
                                            "the cost operator {}"
                                            .format([_.num_qubits for _ in self.mixer_operator[mixer_qubit_check]],
                                                     mixer_qubit_check[0], self.num_qubits)) 
            else:
                if self.mixer_operator.num_qubits != self.num_qubits:
                    valid = False
                    if raise_on_failure:
                        raise AttributeError(
                            "The number of qubits of the mixer {} does not match "
                            "the number of qubits of the cost operator {}".format(
                                self.mixer_operator.num_qubits, self.num_qubits
                            )
                        )

        return valid

    def _build(self):
    
        if self._data is not None:
            return
        self._check_configuration()
        self._data = []
        def build_ansatz_circuit(operators_): # builds the ansatz from the list of mixer operators
            circuits = []
            is_evolved_operator = []
            coeff = Parameter("c")
            for op in operators_:
                if isinstance(op, QuantumCircuit):
                    circuits.append(op)
                    is_evolved_operator.append(False)  # has no time coeff
                else:
                    # check if the operator is just the identity, if yes, skip it
                    if isinstance(op, PauliOp):
                        # possibly need a replacement for the numpy import here
                        # TODO
                        sig_qubits = np.logical_or(op.primitive.x, op.primitive.z)
                        if sum(sig_qubits) == 0:
                            continue
                    evolved_op = self.evolution.convert((coeff*op).exp_i()).reduce()  # ------------ check this, might need negative?
                    circuits.append(evolved_op.to_circuit())
                    is_evolved_operator.append(True)  # has time coeff

            num_qubits = circuits[0].num_qubits
            try:
                qr = QuantumRegister(num_qubits, "q")
                self.add_register(qr)
            except CircuitError:
                # the register already exists, probably because of a previous composition
                pass         
            #times = ParameterVector("t", self.reps * sum(is_evolved_operator))
            evolution_ = QuantumCircuit(*self.qregs, name=self.name)            # ------- need to figure out how initial_point is passed/ updated with reps
            first = True
            for _ in range(self.reps):
                for is_evolved, circuit in zip(is_evolved_operator, circuits):
                    if first:
                        first = False
                    else:
                        if self._insert_barriers:
                            evolution_.barrier()

                    if is_evolved:
                        bound = circuit.assign_parameters({circuit.parameter})
                    else:
                        bound = circuit
                    evolution_.compose(bound, inplace=True)
                # then append opt params to self.gamma_values and self.beta_values
            return evolution_
        varied_operators = list(itertools.chain.from_iterable([[self.operators[0],_] for _ in self.operators[-1]]))
        evolution = build_ansatz_circuit(varied_operators)
        try:
            instr = evolution.to_gate()
        except QiskitError:
            instr = evolution.to_instruction()
        
        self.append(instr, self.qubits)

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits, specified by the size of the cost operator."""
        if self.cost_operator is not None:
            return self.cost_operator.num_qubits
        return 0

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
        if self.num_qubits > 0:
            # local imports to avoid circular imports
            from qiskit.opflow import I, X

            # Mixer is just a sum of single qubit X's on each qubit. Evolving by this operator
            # will simply produce rx's on each qubit.
            mixer_terms = [
                (I ^ left) ^ X ^ (I ^ (self.num_qubits - left - 1))
                for left in range(self.num_qubits)
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
