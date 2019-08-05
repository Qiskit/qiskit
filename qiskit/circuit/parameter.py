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

import sympy

from .parameterexpression import ParameterExpression


class Parameter(ParameterExpression):
    """Parameter Class for variable parameters"""
    def __init__(self, name):
        self._name = name

        symbol = sympy.Symbol(name)
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
