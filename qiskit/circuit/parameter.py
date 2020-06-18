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
"""
Parameter Class for variable parameters.
"""

from uuid import uuid4

from .parameterexpression import ParameterExpression


class Parameter(ParameterExpression):
    """Parameter Class for variable parameters."""

    def __new__(cls, name, uuid=None):  # pylint:disable=unused-argument
        # Parameter relies on self._uuid being set prior to other attributes
        # (e.g. symbol_map) which may depend on self._uuid for Parameter's hash
        # or __eq__ functions.
        obj = object.__new__(cls)

        if uuid is None:
            obj._uuid = uuid4()
        else:
            obj._uuid = uuid

        obj._hash = hash(obj._uuid)
        return obj

    def __getnewargs__(self):
        # Unpickling won't in general call __init__ but will always call
        # __new__. Specify arguments to be passed to __new__ when unpickling.

        return (self.name, self._uuid)

    def __init__(self, name):
        self._name = name

        from sympy import Symbol
        symbol = Symbol(name)
        super().__init__(symbol_map={self: symbol}, expr=symbol)

    def subs(self, parameter_map):
        """Substitute self with the corresponding parameter in parameter_map."""
        return parameter_map[self]

    @property
    def name(self):
        """Returns the name of the Parameter."""
        return self._name

    def __str__(self):
        return self.name

    def __copy__(self):
        return self

    def __deepcopy__(self, memo=None):
        return self

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.name)

    def __eq__(self, other):
        if isinstance(other, Parameter):
            return self._uuid == other._uuid
        elif isinstance(other, ParameterExpression):
            return super().__eq__(other)
        else:
            return False

    def __hash__(self):
        return self._hash
