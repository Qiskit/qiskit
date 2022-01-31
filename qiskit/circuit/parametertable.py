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
import warnings
import functools
from collections.abc import MutableMapping, MappingView

from .instruction import Instruction


class ParameterTable(MutableMapping):
    """Class for managing and setting circuit parameters"""

    __slots__ = ["_table", "_keys", "_names"]

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
        return self._keys

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
        return f"ParameterTable({repr(self._table)})"


def _deprecated_set_method():
    def deprecate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # warn only once
            if not wrapper._warned:
                warnings.warn(
                    f"The ParameterView.{func.__name__} method is deprecated as of "
                    "Qiskit Terra 0.17.0 and will be removed no sooner than 3 months "
                    "after the release date. Circuit parameters are returned as View "
                    "object, not set. To use set methods you can explicitly cast to a "
                    "set.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                wrapper._warned = True
            return func(*args, **kwargs)

        wrapper._warned = False
        return wrapper

    return deprecate


class ParameterView(MappingView):
    """Temporary class to transition from a set return-type to list.

    Derives from a list but implements all set methods, but all set-methods emit deprecation
    warnings.
    """

    def __init__(self, iterable=None):
        if iterable is not None:
            self.data = list(iterable)
        else:
            self.data = []

        super().__init__(self.data)

    @_deprecated_set_method()
    def add(self, x):
        """Add a new element."""
        if x not in self.data:
            self.data.append(x)

    def copy(self):
        """Copy the ParameterView."""
        return self.__class__(self.data.copy())

    @_deprecated_set_method()
    def difference(self, *s):
        """Get the difference between self and the input."""
        return self.__sub__(s)

    @_deprecated_set_method()
    def difference_update(self, *s):
        """Get the difference between self and the input in-place."""
        for element in self:
            if element in s:
                self.remove(element)

    @_deprecated_set_method()
    def discard(self, x):
        """Remove an element from self."""
        if x in self:
            self.remove(x)

    @_deprecated_set_method()
    def intersection(self, *x):
        """Get the intersection between self and the input."""
        return self.__and__(x)

    @_deprecated_set_method()
    def intersection_update(self, *x):
        """Get the intersection between self and the input in-place."""
        return self.__iand__(x)

    def isdisjoint(self, x):
        """Check whether self and the input are disjoint."""
        return not any(element in self for element in x)

    @_deprecated_set_method()
    def issubset(self, x):
        """Check whether self is a subset of the input."""
        return self.__le__(x)

    @_deprecated_set_method()
    def issuperset(self, x):
        """Check whether self is a superset of the input."""
        return self.__ge__(x)

    @_deprecated_set_method()
    def symmetric_difference(self, x):
        """Get the symmetric difference of self and the input."""
        return self.__xor__(x)

    @_deprecated_set_method()
    def symmetric_difference_update(self, x):
        """Get the symmetric difference of self and the input in-place."""
        backward = x.difference(self)
        self.difference_update(x)
        self.update(backward)

    @_deprecated_set_method()
    def union(self, *x):
        """Get the union of self and the input."""
        return self.__or__(x)

    @_deprecated_set_method()
    def update(self, *x):
        """Update self with the input."""
        for element in x:
            self.add(element)

    def remove(self, x):
        """Remove an existing element from the view."""
        self.data.remove(x)

    def __repr__(self):
        """Format the class as string."""
        return f"ParameterView({self.data})"

    def __getitem__(self, index):
        """Get items."""
        return self.data[index]

    def __and__(self, x):
        """Get the intersection between self and the input."""
        inter = []
        for element in self:
            if element in x:
                inter.append(element)

        return self.__class__(inter)

    def __rand__(self, x):
        """Get the intersection between self and the input."""
        return self.__and__(x)

    def __iand__(self, x):
        """Get the intersection between self and the input in-place."""
        for element in self:
            if element not in x:
                self.remove(element)
        return self

    def __len__(self):
        """Get the length."""
        return len(self.data)

    def __or__(self, x):
        """Get the union of self and the input."""
        return set(self) | set(x)

    def __ior__(self, x):
        """Update self with the input."""
        self.update(*x)
        return self

    def __sub__(self, x):
        """Get the difference between self and the input."""
        return set(self) - set(x)

    @_deprecated_set_method()
    def __isub__(self, x):
        """Get the difference between self and the input in-place."""
        return self.difference_update(*x)

    def __xor__(self, x):
        """Get the symmetric difference between self and the input."""
        return set(self) ^ set(x)

    @_deprecated_set_method()
    def __ixor__(self, x):
        """Get the symmetric difference between self and the input in-place."""
        self.symmetric_difference_update(x)
        return self

    def __ne__(self, other):
        return set(other) != set(self)

    def __eq__(self, other):
        return set(other) == set(self)

    def __le__(self, x):
        return all(element in x for element in self)

    def __lt__(self, x):
        if x != self:
            return self <= x
        return False

    def __ge__(self, x):
        return all(element in self for element in x)

    def __gt__(self, x):
        if x != self:
            return self >= x
        return False

    def __iter__(self):
        return iter(self.data)

    def __contains__(self, x):
        return x in self.data

    __hash__: None  # type: ignore
    __ror__ = __or__
