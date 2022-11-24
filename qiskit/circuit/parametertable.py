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
from collections.abc import MappingView, MutableMapping, MutableSet


class ParameterReferences(MutableSet):
    """A set of instruction parameter slot references.
    Items are expected in the form ``(instruction, param_index)``. Membership
    testing is overriden such that items that are otherwise value-wise equal
    are still considered distinct if their ``instruction``\\ s are referentially
    distinct.
    """

    def _instance_key(self, ref):
        return (id(ref[0]), ref[1])

    def __init__(self, refs):
        self._instance_ids = {}

        for ref in refs:
            if not isinstance(ref, tuple) or len(ref) != 2:
                raise ValueError("refs must be in form (instruction, param_index)")
            k = self._instance_key(ref)
            self._instance_ids[k] = ref[0]

    def __getstate__(self):
        # Leave behind the reference IDs (keys of _instance_ids) since they'll
        # be incorrect after unpickling on the other side.
        return list(self)

    def __setstate__(self, refs):
        # Recompute reference IDs for the newly unpickled instructions.
        self._instance_ids = {self._instance_key(ref): ref[0] for ref in refs}

    def __len__(self):
        return len(self._instance_ids)

    def __iter__(self):
        for (_, idx), instruction in self._instance_ids.items():
            yield (instruction, idx)

    def __contains__(self, x) -> bool:
        return self._instance_key(x) in self._instance_ids

    def __repr__(self) -> str:
        return f"ParameterReferences({repr(list(self))})"

    def add(self, value):
        """Adds a reference to the listing if it's not already present."""
        k = self._instance_key(value)
        self._instance_ids[k] = value[0]

    def discard(self, value):
        k = self._instance_key(value)
        self._instance_ids.pop(k, None)

    def copy(self):
        """Create a shallow copy."""
        return ParameterReferences(self)


class ParameterTable(MutableMapping):
    """Class for tracking references to circuit parameters by specific
    instruction instances.

    Keys are parameters. Values are of type :class:`~ParameterReferences`,
    which overrides membership testing to be referential for instructions,
    and is set-like. Elements of :class:`~ParameterReferences`
    are tuples of ``(instruction, param_index)``.
    """

    __slots__ = ["_table", "_keys", "_names"]

    def __init__(self, mapping=None):
        """Create a new instance, initialized with ``mapping`` if provided.

        Args:
            mapping (Mapping[Parameter, ParameterReferences]):
                Mapping of parameter to the set of parameter slots that reference
                it.

        Raises:
            ValueError: A value in ``mapping`` is not a :class:`~ParameterReferences`.
        """
        if mapping is not None:
            if any(not isinstance(refs, ParameterReferences) for refs in mapping.values()):
                raise ValueError("Values must be of type ParameterReferences")
            self._table = mapping.copy()
        else:
            self._table = {}

        self._keys = set(self._table)
        self._names = {x.name for x in self._table}

    def __getitem__(self, key):
        return self._table[key]

    def __setitem__(self, parameter, refs):
        """Associate a parameter with the set of parameter slots ``(instruction, param_index)``
        that reference it.

        .. note::

            Items in ``refs`` are considered unique if their ``instruction`` is referentially
            unique. See :class:`~ParameterReferences` for details.

        Args:
            parameter (Parameter): the parameter
            refs (Union[ParameterReferences, Iterable[(Instruction, int)]]): the parameter slots.
                If this is an iterable, a new :class:`~ParameterReferences` is created from its
                contents.
        """
        if not isinstance(refs, ParameterReferences):
            refs = ParameterReferences(refs)

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

    def copy(self):
        """Copy the ParameterView."""
        return self.__class__(self.data.copy())

    def isdisjoint(self, x):
        """Check whether self and the input are disjoint."""
        return not any(element in self for element in x)

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

    def __sub__(self, x):
        """Get the difference between self and the input."""
        return set(self) - set(x)

    def __xor__(self, x):
        """Get the symmetric difference between self and the input."""
        return set(self) ^ set(x)

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
