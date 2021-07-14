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

"""The evolved operator ansatz."""

from typing import Optional

import numpy as np

from qiskit.circuit import Parameter, ParameterVector, QuantumRegister, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.exceptions import QiskitError

from .blueprintcircuit import BlueprintCircuit


class EvolvedOperatorAnsatz(BlueprintCircuit):
    """The evolved operator ansatz."""

    def __init__(
        self,
        operators=None,
        reps: int = 1,
        evolution=None,
        insert_barriers: bool = False,
        name: str = "EvolvedOps",
        initial_state: Optional[QuantumCircuit] = None,
    ):
        """
        Args:
            operators (Optional[Union[OperatorBase, QuantumCircuit, list]): The operators to evolve.
                If a circuit is passed, we assume it implements an already evolved operator and thus
                the circuit is not evolved again. Can be a single operator (circuit) or a list of
                operators (and circuits).
            reps: The number of times to repeat the evolved operators.
            evolution (Optional[EvolutionBase]): An opflow converter object to construct the evolution.
                Defaults to Trotterization.
            insert_barriers: Whether to insert barriers in between each evolution.
            name: The name of the circuit.
            initial_state: A `QuantumCircuit` object to prepend to the circuit.
        """
        if evolution is None:
            # pylint: disable=cyclic-import
            from qiskit.opflow import PauliTrotterEvolution

            evolution = PauliTrotterEvolution()

        if operators is not None:
            operators = _validate_operators(operators)

        super().__init__(name=name)
        self._operators = operators
        self._evolution = evolution
        self._reps = reps
        self._insert_barriers = insert_barriers
        self._initial_state = initial_state

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        if self.operators is None:
            if raise_on_failure:
                raise ValueError("The operators are not set.")
            return False

        if self.reps < 1:
            if raise_on_failure:
                raise ValueError("The reps cannot be smaller than 1.")
            return False

        return True

    @property
    def reps(self) -> int:
        """The number of times the evolved operators are repeated."""
        return self._reps

    @reps.setter
    def reps(self, r: int) -> None:
        """Sets the number of times the evolved operators are repeated."""
        self._invalidate()
        self._reps = r

    @property
    def evolution(self):
        """The evolution converter used to compute the evolution.

        Returns:
            EvolutionBase: The evolution converter used to compute the evolution.
        """
        return self._evolution

    @evolution.setter
    def evolution(self, evol) -> None:
        """Sets the evolution converter used to compute the evolution.

        Args:
            evol (EvolutionBase): An opflow converter object to construct the evolution.
        """
        self._invalidate()
        self._evolution = evol

    @property
    def initial_state(self) -> QuantumCircuit:
        """The initial state."""
        return self._initial_state

    @initial_state.setter
    def initial_state(self, initial_state: QuantumCircuit) -> None:
        """Sets the initial state."""
        self._invalidate()
        self._initial_state = initial_state

    @property
    def operators(self):
        """The operators that are evolved in this circuit.

        Returns:
            list: The operators to be evolved (and circuits) contained in this ansatz.
        """
        return self._operators

    @operators.setter
    def operators(self, operators=None) -> None:
        """Set the operators to be evolved.

        operators (Optional[Union[OperatorBase, QuantumCircuit, list]): The operators to evolve.
            If a circuit is passed, we assume it implements an already evolved operator and thus
            the circuit is not evolved again. Can be a single operator (circuit) or a list of
            operators (and circuits).
        """
        operators = _validate_operators(operators)
        self._invalidate()
        self._operators = operators

    @property
    def qregs(self):
        """A list of the quantum registers associated with the circuit."""
        if self._data is None:
            self._build()
        return self._qregs

    @qregs.setter
    def qregs(self, qregs):
        """Set the quantum registers associated with the circuit."""
        self._qregs = qregs
        self._qubits = [qbit for qreg in qregs for qbit in qreg]
        self._invalidate()

    # TODO: the `preferred_init_points`-implementation can (and should!) be improved!
    @property
    def preferred_init_points(self):
        """Getter of preferred initial points based on the given initial state."""
        if self._initial_state is None:
            return None
        else:
            # If an initial state was set by the user, then we want to make sure that the VQE does
            # not start from a random point. Thus, we return an all-zero initial point for the
            # optimizer which is used (unless it gets overwritten by a higher-priority setting at
            # runtime of the VQE).
            # However, in order to determine the correct length, we must build the QuantumCircuit
            # first, because otherwise the operators may not be set yet.
            self._build()
            return np.zeros(self.reps * len(self.operators), dtype=float)

    def _build(self):
        if self._data is not None:
            return

        self._check_configuration()
        self._data = []

        # get the evolved operators as circuits
        from qiskit.opflow import PauliOp

        coeff = Parameter("c")
        circuits = []
        is_evolved_operator = []
        for op in self.operators:
            # if the operator is already the evolved circuit just append it
            if isinstance(op, QuantumCircuit):
                circuits.append(op)
                is_evolved_operator.append(False)  # has no time coeff
            else:
                # check if the operator is just the identity, if yes, skip it
                if isinstance(op, PauliOp):
                    sig_qubits = np.logical_or(op.primitive.x, op.primitive.z)
                    if sum(sig_qubits) == 0:
                        continue

                evolved_op = self.evolution.convert((coeff * op).exp_i()).reduce()
                circuits.append(evolved_op.to_circuit())
                is_evolved_operator.append(True)  # has time coeff

        # set the registers
        num_qubits = circuits[0].num_qubits
        try:
            qr = QuantumRegister(num_qubits, "q")
            self.add_register(qr)
        except CircuitError:
            # the register already exists, probably because of a previous composition
            pass

        # build the circuit
        times = ParameterVector("t", self.reps * sum(is_evolved_operator))
        times_it = iter(times)

        evolution = QuantumCircuit(*self.qregs, name=self.name)

        first = True
        for _ in range(self.reps):
            for is_evolved, circuit in zip(is_evolved_operator, circuits):
                if first:
                    first = False
                else:
                    if self._insert_barriers:
                        evolution.barrier()

                if is_evolved:
                    bound = circuit.assign_parameters({coeff: next(times_it)})
                else:
                    bound = circuit

                evolution.compose(bound, inplace=True)

        if self.initial_state:
            evolution.compose(self.initial_state, front=True, inplace=True)

        try:
            instr = evolution.to_gate()
        except QiskitError:
            instr = evolution.to_instruction()

        self.append(instr, self.qubits)


def _validate_operators(operators):
    if not isinstance(operators, list):
        operators = [operators]

    if len(operators) > 1:
        num_qubits = operators[0].num_qubits
        if any(operators[i].num_qubits != num_qubits for i in range(1, len(operators))):
            raise ValueError("All operators must act on the same number of qubits.")

    return operators
