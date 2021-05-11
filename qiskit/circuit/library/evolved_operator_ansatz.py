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

from typing import List, Union, Optional

import numpy as np

from qiskit.circuit import Parameter, ParameterVector, QuantumRegister, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError

from .blueprintcircuit import BlueprintCircuit


class EvolvedOperatorAnsatz(BlueprintCircuit):
    """The evolved operator ansatz."""

    def __init__(
        self,
        operators: Optional[Union["OperatorBase", List["OperatorBase"]]] = None,
        reps: int = 1,
        evolution: Optional["EvolutionBase"] = None,
        insert_barriers: bool = False,
        name: str = "EvolvedOps",
        initial_state: Optional[QuantumCircuit] = None,
    ):
        """
        Args:
            operators: The operators to evolve.
            reps: The number of times to repeat the evolved operators.
            evolution: An opflow converter object to construct the evolution.
                Defaults to Trotterization.
            insert_barriers: Whether to insert barriers in between each evolution.
            name: The name of the circuit.
            initial_state: A `QuantumCircuit` object to prepend to the circuit.
        """
        if evolution is None:
            from qiskit.opflow import PauliTrotterEvolution

            evolution = PauliTrotterEvolution()

        super().__init__(name=name)
        self._operators = None
        self._evolution = evolution
        self._reps = reps
        self._insert_barriers = insert_barriers
        self._initial_state = initial_state

        # use setter to set operators
        self.operators = operators

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
    def evolution(self) -> "EvolutionBase":
        """The evolution converter used to compute the evolution."""
        return self._evolution

    @evolution.setter
    def evolution(self, evol: "EvolutionBase"):
        """Sets the evolution converter used to compute the evolution."""
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
    def operators(self) -> List["OperatorBase"]:
        """The operators that are evolved in this circuit."""
        return self._operators

    @operators.setter
    def operators(self, operators: Union["OperatorBase", List["OperatorBase"]]) -> None:
        """Set the operators to be evolved."""
        if not isinstance(operators, list):
            operators = [operators]

        if len(operators) > 1:
            num_qubits = operators[0].num_qubits
            if any(operators[i].num_qubits != num_qubits for i in range(1, len(operators))):
                raise ValueError("All operators must act on the same number of qubits.")

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
        coeff = Parameter("c")
        evolved_ops = [self.evolution.convert((coeff * op).exp_i()) for op in self.operators]
        circuits = [evolved_op.reduce().to_circuit() for evolved_op in evolved_ops]

        # set the registers
        num_qubits = circuits[0].num_qubits
        try:
            qr = QuantumRegister(num_qubits, "q")
            self.add_register(qr)
        except CircuitError:
            # the register already exists, probably because of a previous composition
            pass

        # build the circuit
        times = ParameterVector("t", self.reps * len(self.operators))
        times_it = iter(times)

        first = True
        for _ in range(self.reps):
            for circuit in circuits:
                if first:
                    first = False
                else:
                    if self._insert_barriers:
                        self.barrier()
                self.compose(circuit.assign_parameters({coeff: next(times_it)}), inplace=True)

        if self._initial_state:
            self.compose(self._initial_state, front=True, inplace=True)
