# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Look-up table for variable parameters in QuantumCircuit.
"""
from collections.abc import MutableMapping, MappingView

from .instruction import Instruction


class ParameterTable(MutableMapping):
    """Class for managing and setting circuit parameters"""

    __slots__ = ['_table', '_keys', '_names']

    def __init__(self, *args, **kwargs):
        """
        the structure of _table is,
           {var_object: [(instruction_object, parameter_index), ...]}
        """
        self._table = dict(*args, **kwargs)
        self._keys = set(self._table)
        self._names = {x.name for x in self._table}

    def __getitem__(self, key):
        return self._table[key]

    def __setitem__(self, parameter, instr_params):
        """Sets list of Instructions that depend on Parameter.

        Args:
            parameter (Parameter): the parameter to set
            instr_params (list): List of (Instruction, int) tuples. Int is the
              parameter index at which the parameter appears in the instruction.
        """

        for instruction, param_index in instr_params:
            assert isinstance(instruction, Instruction)
            assert isinstance(param_index, int)
        self._table[parameter] = instr_params
        self._keys.add(parameter)
        self._names.add(parameter.name)

    def get_keys(self):
        """Return a set of all keys in the parameter table

        Returns:
            set: A set of all the keys in the parameter table
        """
        return ParameterView(self._table.keys())

    def get_names(self):
        """Return a set of all parameter names in the parameter table

        Returns:
            set: A set of all the names in the parameter table
        """
        return self._names

    def __delitem__(self, key):
        del self._table[key]
        self._keys.discard(key)
        self._names.discard(key.name)

    def __iter__(self):
        return iter(self._table)

    def __len__(self):
        return len(self._table)

    def __repr__(self):
        return 'ParameterTable({})'.format(repr(self._table))


# m = MutableSet()
# m.add

# pylint: disable=invalid-name


class ParameterView(MappingView):
    """Temporary class to transition from a set return-type to list.

    Derives from a list but implements all set methods, but all set-methods emit deprecation
    warnings.
    """

    def __init__(self, iterable=None):
        super().__init__(iterable)

        self.data = []

        if iterable is not None:
            for x in iterable:
                self.add(x)

    def add(self, x):
        """Add a new element."""
        if x not in self.data:
            self.data.append(x)

    def copy(self):
        """Copy the ParameterView."""
        return self.__class__(self.data.copy())

    def difference(self, *s):
        """Get the difference between self and the input."""
        diff = self.__class__()
        for element in self:
            if element not in s:
                diff.add(element)

        return diff

    def difference_update(self, *s):
        """Get the difference between self and the input in-place."""
        for element in self:
            if element in s:
                self.remove(element)

    def discard(self, x):
        """Remove an element from self."""
        if x in self:
            self.remove(x)

    def intersection(self, *s):
        """Get the intersection between self and the input."""
        inter = self.__class__()
        for element in self:
            if element in s:
                inter.add(element)

        return inter

    def intersection_update(self, *s):
        """Get the intersection between self and the input in-place."""
        for element in self:
            if element not in s:
                self.remove(element)

    def isdisjoint(self, s):
        """Check whether self and the input are disjoint."""
        return not any(element in self for element in s)

    def issubset(self, s):
        """Check whether self is a subset of the input."""
        return all(element in s for element in self)

    def issuperset(self, s):
        """Check whether self is a superset of the input."""
        return all(element in self for element in s)

    def symmetric_difference(self, s):
        """Get the symmetric difference of self and the input."""
        forward = self.difference(s)
        backward = s.difference(self)
        return forward.update(backward)

    def symmetric_difference_update(self, s):
        """Get the symmetric difference of self and the input in-place."""
        backward = s.difference(self)
        self.difference_update(s)
        self.update(backward)

    def union(self, *s):
        """Get the union of self and the input."""
        joint = self.copy()
        joint.update(s)
        return joint

    def update(self, *s):
        """Update self with the input."""
        for element in s:
            self.add(element)

    def remove(self, x):
        """Remove an existing element from the view."""
        self.data.remove(x)

    def __and__(self, s):
        """Get the intersection between self and the input."""
        return self.intersection(s)

    def __iand__(self, s):
        """Get the intersection between self and the input in-place."""
        self.intersection_update(s)
        return self

    def __len__(self):
        """Get the length."""
        return len(self.data)

    def __or__(self, s):
        """Get the union of self and the input."""
        return self.union(s)

    def __ior__(self, s):
        """Update self with the input."""
        self.update(s)
        return self

    def __sub__(self, s):
        """Get the difference between self and the input."""
        return self.difference(s)

    def __isub__(self, s):
        """Get the difference between self and the input in-place."""
        return self.difference_update(s)

    def __xor__(self, s):
        """Get the symmetric difference between self and the input."""
        return self.symmetric_difference(s)

    def __ixor__(self, s):
        """Get the symmetric difference between self and the input in-place."""
        self.symmetric_difference_update(s)
        return self

    def __ne__(self, other):
        return set(other) != set(self)

    def __eq__(self, other):
        return set(other) == set(self)

    def __le__(self, s):
        return self.issubset(s)

    def __lt__(self, s):
        if s != self:
            return self <= s
        return False

    def __ge__(self, s):
        return self.issuperset(s)

    def __gt__(self, s):
        if s != self:
            return self >= s
        return False

    def __iter__(self):
        return iter(self.data)

    def __contains__(self, o):
        return o in self.data

    __hash__: None  # type: ignore
