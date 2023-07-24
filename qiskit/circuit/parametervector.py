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

"""Parameter Vector Class to simplify management of parameter lists."""

from uuid import uuid4, UUID

from .parameter import Parameter


class ParameterVectorElement(Parameter):
    """An element of a ParameterVector."""

    ___slots__ = ("_vector", "_index")

    def __new__(cls, vector, index, uuid=None):  # pylint:disable=unused-argument
        obj = object.__new__(cls)

        if uuid is None:
            obj._uuid = uuid4()
        else:
            obj._uuid = uuid

        obj._hash = hash(obj._uuid)
        return obj

    def __getnewargs__(self):
        return (self.vector, self.index, self._uuid)

    def __init__(self, vector, index, uuid=None):  # pylint: disable=unused-argument
        name = f"{vector.name}[{index}]"
        super().__init__(name)
        self._vector = vector
        self._index = index

    @property
    def index(self):
        """Get the index of this element in the parent vector."""
        return self._index

    @property
    def vector(self):
        """Get the parent vector instance."""
        return self._vector

    def __getstate__(self):
        return {
            "name": self._name,
            "uuid": self._uuid,
            "vector": self._vector,
            "index": self._index,
        }

    def __setstate__(self, state):
        self._name = state["name"]
        self._uuid = state["uuid"]
        self._vector = state["vector"]
        self._index = state["index"]
        super().__init__(self._name)


class ParameterVector:
    """ParameterVector class to quickly generate lists of parameters."""

    __slots__ = ("_name", "_params", "_size", "_root_uuid")

    def __init__(self, name, length=0):
        self._name = name
        self._size = length
        self._root_uuid = uuid4()
        root_uuid_int = self._root_uuid.int
        self._params = [
            ParameterVectorElement(self, i, UUID(int=root_uuid_int + i)) for i in range(length)
        ]

    @property
    def name(self):
        """Returns the name of the ParameterVector."""
        return self._name

    @property
    def params(self):
        """Returns the list of parameters in the ParameterVector."""
        return self._params

    def index(self, value):
        """Returns first index of value."""
        return self._params.index(value)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self._size)
            return self.params[start:stop:step]

        if key > self._size:
            raise IndexError(f"Index out of range: {key} > {self._size}")
        return self.params[key]

    def __iter__(self):
        return iter(self.params[: self._size])

    def __len__(self):
        return self._size

    def __str__(self):
        return f"{self.name}, {[str(item) for item in self.params[: self._size]]}"

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, length={len(self)})"

    def resize(self, length):
        """Resize the parameter vector.

        If necessary, new elements are generated. If length is smaller than before, the
        previous elements are cached and not re-generated if the vector is enlarged again.
        This is to ensure that the parameter instances do not change.
        """
        if length > len(self._params):
            root_uuid_int = self._root_uuid.int
            self._params.extend(
                [
                    ParameterVectorElement(self, i, UUID(int=root_uuid_int + i))
                    for i in range(len(self._params), length)
                ]
            )
        self._size = length
