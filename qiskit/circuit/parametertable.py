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
import functools
import warnings
from collections.abc import MappingView, MutableMapping, Set
from typing import MutableSequence


class ParameterReferences(MutableSequence, Set):
    """An (insertion) ordered set of instruction parameter slot references.
    Items are expected in the form ``(instruction, param_index)``. Membership
    testing is overriden such that items that are otherwise value-wise equal
    are still considered distinct if their ``instruction``\ s are referentially
    distinct.

    Items are ordered by insertion, and may be read or deleted by sequence index.
    However, update by index and insert by index are not supported, since
    these actions inherently violate insertion order.
    """

    def _instance_key(self, ref):
        return (id(ref[0]), ref[1])

    def __init__(self, *refs) -> None:
        self._data = []
        self._instance_ids = set()

        for ref in refs:
            k = self._instance_key(ref)
            if k not in self._instance_ids:
                self._data.append(ref)
                self._instance_ids.add(k)

    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, ref):
        raise NotImplementedError("Position is dictated by insertion order.")

    def __delitem__(self, index):
        ref = self._data[index]
        del self._data[index]
        self._instance_ids.discard(self._instance_key(ref))

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __contains__(self, x) -> bool:
        return self._instance_key(x) in self._instance_ids

    def __repr__(self) -> str:
        return repr(self._data)

    def insert(self, index, ref) -> None:
        raise NotImplementedError("Position is dictated by insertion order.")

    def append(self, ref):
        """Adds a reference to the listing if it's not already present."""
        k = self._instance_key(ref)
        if k in self._instance_ids:
            return

        self._data.append(ref)
        self._instance_ids.add(k)

    def copy(self):
        return ParameterReferences(self._data)


class ParameterTable(MutableMapping):
    """Class for tracking references to circuit parameters by specific
    instruction instances.

    Keys are parameters. Values are of type :class:`~ParameterReferences`,
    which overrides membership testing to be referential for instructions,
    and is both set-like and sequence-like. Elements of :class:`~ParameterReferences`
    are tuples of ``(instruction, param_index)``.
    """

    __slots__ = ["_table", "_keys", "_names"]

    def __init__(self, mapping=None):
        """Create a new instance, initialized with ``mapping`` if provided.

        Args:
            mapping (Mapping[Parameter, ReferenceListing]):
                Mapping of parameter to the set of parameter slots that reference
                it.
        """
        if mapping is not None:
            if any(not isinstance(ParameterReferences, refs) for refs in mapping.values()):
                raise ValueError("Values must be of type ParameterReferences")
            self._table = {param: refs for param, refs in mapping.items()}
        else:
            self._table = {}

        self._keys = set(self._table)
        self._names = {x.name for x in self._table}

    def __getitem__(self, key):
        return self._table[key]

    def __setitem__(self, parameter, refs):
        """Associate a parameter with the set of parameter slots that reference it.

        .. note::

            Items in ``refs`` are considered unique if their ``instruction`` is referentially
            unique. See :class:`~ParameterReferences` for details.

        Args:
            parameter (Parameter): the parameter.
            refs (ParameterReferences): the parameter slots.
        """
        if not isinstance(refs, ParameterReferences):
            raise ValueError("Value must be of type ParameterReferences")

        self._table[parameter] = refs
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
