# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A wrapper class for the purposes of validating modifications to
QuantumCircuit.data while maintaining the interface of a python list."""

from collections.abc import MutableSequence

from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.instruction import Instruction


class QuantumCircuitData(MutableSequence):
    """A wrapper class for the purposes of validating modifications to
    QuantumCircuit.data while maintaining the interface of a python list."""

    def __init__(self, circuit):
        self._circuit = circuit

    def __getitem__(self, i):
        return self._circuit._data[i]

    def __setitem__(self, key, value):
        instruction, qargs, cargs = value

        if not isinstance(instruction, Instruction) and hasattr(instruction, "to_instruction"):
            instruction = instruction.to_instruction()

        expanded_qargs = [self._circuit.qbit_argument_conversion(qarg) for qarg in qargs or []]
        expanded_cargs = [self._circuit.cbit_argument_conversion(carg) for carg in cargs or []]

        broadcast_args = list(instruction.broadcast_arguments(expanded_qargs, expanded_cargs))

        if len(broadcast_args) > 1:
            raise CircuitError(
                "QuantumCircuit.data modification does not " "support argument broadcasting."
            )

        qargs, cargs = broadcast_args[0]

        if not isinstance(instruction, Instruction):
            raise CircuitError("object is not an Instruction.")

        self._circuit._check_dups(qargs)
        self._circuit._check_qargs(qargs)
        self._circuit._check_cargs(cargs)

        self._circuit._data[key] = (instruction, qargs, cargs)

        self._circuit._update_parameter_table(instruction)

    def insert(self, index, value):
        self._circuit._data.insert(index, None)
        self[index] = value

    def __delitem__(self, i):
        del self._circuit._data[i]

    def __len__(self):
        return len(self._circuit._data)

    def __cast(self, other):
        return other._circuit._data if isinstance(other, QuantumCircuitData) else other

    def __repr__(self):
        return repr(self._circuit._data)

    def __lt__(self, other):
        return self._circuit._data < self.__cast(other)

    def __le__(self, other):
        return self._circuit._data <= self.__cast(other)

    def __eq__(self, other):
        return self._circuit._data == self.__cast(other)

    def __gt__(self, other):
        return self._circuit._data > self.__cast(other)

    def __ge__(self, other):
        return self._circuit._data >= self.__cast(other)

    def __add__(self, other):
        return self._circuit._data + self.__cast(other)

    def __radd__(self, other):
        return self.__cast(other) + self._circuit._data

    def __mul__(self, n):
        return self._circuit._data * n

    def __rmul__(self, n):
        return n * self._circuit._data

    def sort(self, *args, **kwargs):
        """In-place stable sort. Accepts arguments of list.sort."""
        self._circuit._data.sort(*args, **kwargs)

    def copy(self):
        """Returns a shallow copy of instruction list."""
        return self._circuit._data.copy()
