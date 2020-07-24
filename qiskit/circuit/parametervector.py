# -*- coding: utf-8 -*-

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

from .parameter import Parameter


class ParameterVector:
    """ParameterVector class to quickly generate lists of parameters."""

    def __init__(self, name, length=0):
        self._name = name
        self._params = []
        self._size = length
        for i in range(length):
            self._params += [Parameter('{0}[{1}]'.format(self._name, i))]

    @property
    def name(self):
        """Returns the name of the ParameterVector."""
        return self._name

    @property
    def params(self):
        """Returns the list of parameters in the ParameterVector."""
        return self._params

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self._size)
            return self.params[start:stop:step]

        if key > self._size:
            raise IndexError('Index out of range: {} > {}'.format(key, self._size))
        return self.params[key]

    def __iter__(self):
        return iter(self.params[:self._size])

    def __len__(self):
        return self._size

    def __str__(self):
        return '{}, {}'.format(self.name, [str(item) for item in self.params[:self._size]])

    def __repr__(self):
        return '{}(name={}, length={})'.format(self.__class__.__name__, self.name, len(self))

    def resize(self, length):
        """Resize the parameter vector.

        If necessary, new elements are generated. If length is smaller than before, the
        previous elements are cached and not re-generated if the vector is enlargened again.
        This is to ensure that the parameter instances do not change.
        """
        if length > self._size:
            for i in range(self._size, length):
                self._params += [Parameter('{0}[{1}]'.format(self._name, i))]
        self._size = length
