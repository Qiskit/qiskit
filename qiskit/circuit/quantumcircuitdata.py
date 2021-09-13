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

from abc import abstractmethod
from collections.abc import MutableSequence

from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.instruction import Instruction


class ValidatedMutableSequence(MutableSequence):
    """A wrapper for validating and acting upon modifications to an
    underlying List.

    Intended to be used when an underlying list structure, owned by
    the QuantumCircuit (such as QuantumCircuit._data), is fetched by a
    user (through e.g. the QuantumCircuit.data property) and may be
    subsequently updated, such that updates through the returned
    ValidatedMutableSequence are checked to ensure consistency with
    the QuantumCircuit instance.
    """

    def __init__(self, list_):
        self._list = list_

    def __getitem__(self, i):
        return self._list[i]

    @abstractmethod
    def __delitem__(self, i):
        pass

    @abstractmethod
    def __setitem__(self, index, value):
        pass

    @abstractmethod
    def insert(self, index, value):
        pass

    def __len__(self):
        return len(self._list)

    def __cast(self, other):
        return other._list if isinstance(other, type(self)) else other

    def __repr__(self):
        return repr(self._list)

    def __lt__(self, other):
        return self._list < self.__cast(other)

    def __le__(self, other):
        return self._list <= self.__cast(other)

    def __eq__(self, other):
        return self._list == self.__cast(other)

    def __gt__(self, other):
        return self._list > self.__cast(other)

    def __ge__(self, other):
        return self._list >= self.__cast(other)

    def __add__(self, other):
        return self._list + self.__cast(other)

    def __radd__(self, other):
        return self.__cast(other) + self._list

    def __mul__(self, n):
        return self._list * n

    def __rmul__(self, n):
        return n * self._list

    def sort(self, *args, **kwargs):
        """In-place stable sort. Accepts arguments of list.sort."""
        self._list.sort(*args, **kwargs)

    def copy(self):
        """Returns a shallow copy of instruction list."""
        return self._list.copy()


class QuantumCircuitData(ValidatedMutableSequence):
    """A wrapper class for the purposes of validating modifications to
    QuantumCircuit.data while maintaining the interface of a python list."""

    def __init__(self, circuit):
        self._circuit = circuit
        super().__init__(circuit._data)

    def __setitem__(self, key, value):
        instruction, qargs, cargs = value

        if not isinstance(instruction, Instruction) and hasattr(instruction, "to_instruction"):
            instruction = instruction.to_instruction()

        expanded_qargs = [self._circuit.qbit_argument_conversion(qarg) for qarg in qargs or []]
        expanded_cargs = [self._circuit.cbit_argument_conversion(carg) for carg in cargs or []]

        broadcast_args = list(instruction.broadcast_arguments(expanded_qargs, expanded_cargs))

        if len(broadcast_args) > 1:
            raise CircuitError(
                "QuantumCircuit.data modification does not support argument broadcasting."
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
        try:
            # To consolidate validation to __setitem__, insert None and overwrite.
            self._list.insert(index, None)
            self[index] = value
        except CircuitError as err:
            del self._list[index]
            raise err

    def __delitem__(self, index):
        del self._list[index]


class QuantumCircuitQregs(ValidatedMutableSequence):
    """A wrapper class for the purposes of validating modifications to
    QuantumCircuit.qregs while maintaining the interface of a python list."""

    def __init__(self, circuit):
        self._circuit = circuit
        super().__init__(circuit._qregs)

    def __setitem__(self, key, value):
        self._circuit.add_register(value)
        self._list[key] = value
        del self._list[-1]

    def insert(self, index, value):
        self._circuit.add_register(value)
        self._list.insert(index, self._list[-1])
        del self._list[-1]

    def __delitem__(self, index):
        del self._list[index]

    def reverse(self):
        # Override MutableSequence.reverse which uses repeated pairwise
        # calls to __setitem__ that do not maintain uniqueness of registers
        # in _list throughout, and so raises a "CircuitError: register name
        # ... already exists"
        self._list.reverse()


class QuantumCircuitCregs(ValidatedMutableSequence):
    """A wrapper class for the purposes of validating modifications to
    QuantumCircuit.cregs while maintaining the interface of a python list."""

    def __init__(self, circuit):
        self._circuit = circuit
        super().__init__(circuit._cregs)

    def __setitem__(self, key, value):
        self._circuit.add_register(value)
        self._list[key] = value
        del self._list[-1]

    def insert(self, index, value):
        self._circuit.add_register(value)
        self._list.insert(index, self._list[-1])
        del self._list[-1]

    def __delitem__(self, index):
        del self._list[index]

    def reverse(self):
        # Override MutableSequence.reverse which uses repeated pairwise
        # calls to __setitem__ that do not maintain uniqueness of registers
        # in _list throughout, and so raises a "CircuitError: register name
        # ... already exists"
        self._list.reverse()
