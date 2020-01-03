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
ParameterExpression Class to enable creating simple expressions of Parameters.
"""

import numbers
import operator

import numpy

from qiskit.circuit.exceptions import CircuitError


class ParameterExpression():
    """ParameterExpression class to enable creating expressions of Parameters."""

    def __init__(self, symbol_map, expr):
        """Create a new ParameterExpression.

        Not intended to be called directly, but to be instantiated via operations
        on other Parameter or ParameterExpression objects.

        Args:
            symbol_map (dict): Mapping of Parameter instances to the sympy.Symbol
                               serving as their placeholder in expr.
            expr (sympy.Expr): Expression of sympy.Symbols.
        """
        self._parameter_symbols = symbol_map
        self._symbol_expr = expr

    @property
    def parameters(self):
        """Returns a set of the unbound Parameters in the expression."""
        return set(self._parameter_symbols.keys())

    def bind(self, parameter_values):
        """Binds the provided set of parameters to their corresponding values.

        Args:
            parameter_values (dict): Mapping of Parameter instances to the
                                     numeric value to which they will be bound.

        Raises:
            CircuitError:
                - If parameter_values contains Parameters outside those in self.
                - If a non-numeric value is passed in parameter_values.
            ZeroDivisionError:
                - If binding the provided values requires division by zero.

        Returns:
            ParameterExpression: a new expression parameterized by any parameters
                which were not bound by parameter_values.
        """

        self._raise_if_passed_unknown_parameters(parameter_values.keys())
        self._raise_if_passed_non_real_value(parameter_values)

        symbol_values = {self._parameter_symbols[parameter]: value
                         for parameter, value in parameter_values.items()}
        bound_symbol_expr = self._symbol_expr.subs(symbol_values)

        # Don't use sympy.free_symbols to count remaining parameters here.
        # sympy will in some cases reduce the expression and remove even
        # unbound symbols.
        # e.g. (sympy.Symbol('s') * 0).free_symbols == set()

        free_parameters = self.parameters - parameter_values.keys()
        free_parameter_symbols = {p: s for p, s in self._parameter_symbols.items()
                                  if p in free_parameters}

        if bound_symbol_expr.is_infinite:
            raise ZeroDivisionError('Binding provided for expression '
                                    'results in division by zero '
                                    '(Expression: {}, Bindings: {}).'.format(
                                        self, parameter_values))

        return ParameterExpression(free_parameter_symbols, bound_symbol_expr)

    def subs(self, parameter_map):
        """Returns a new Expression with replacement Parameters.

        Args:
            parameter_map (dict): Mapping from Parameters in self to the
                                  Parameter instances with which they should be
                                  replaced.

        Raises:
            CircuitError:
                - If parameter_map contains Parameters outside those in self.
                - If the replacement Parameters in parameter_map would result in
                  a name conflict in the generated expression.

        Returns:
            ParameterExpression: a new expression with the specified parameters
                                 replaced.
        """

        self._raise_if_passed_unknown_parameters(parameter_map.keys())
        self._raise_if_parameter_names_conflict(parameter_map.keys())

        from sympy import Symbol
        new_parameter_symbols = {p: Symbol(p.name)
                                 for p in parameter_map.values()}

        # Include existing parameters in self not set to be replaced.
        new_parameter_symbols.update({p: s
                                      for p, s in self._parameter_symbols.items()
                                      if p not in parameter_map})

        symbol_map = {
            self._parameter_symbols[old_param]: new_parameter_symbols[new_param]
            for old_param, new_param in parameter_map.items()
        }

        substituted_symbol_expr = self._symbol_expr.subs(symbol_map)

        return ParameterExpression(new_parameter_symbols, substituted_symbol_expr)

    def _raise_if_passed_unknown_parameters(self, parameters):
        unknown_parameters = parameters - self.parameters
        if unknown_parameters:
            raise CircuitError('Cannot bind Parameters ({}) not present in '
                               'expression.'.format([str(p) for p in unknown_parameters]))

    def _raise_if_passed_non_real_value(self, parameter_values):
        nonreal_parameter_values = {p: v for p, v in parameter_values.items()
                                    if not isinstance(v, numbers.Real)}
        if nonreal_parameter_values:
            raise CircuitError('Expression cannot bind non-real or non-numeric '
                               'values ({}).'.format(nonreal_parameter_values))

    def _raise_if_parameter_names_conflict(self, other_parameters):
        self_names = {p.name: p for p in self.parameters}
        other_names = {p.name: p for p in other_parameters}

        shared_names = self_names.keys() & other_names.keys()
        conflicting_names = {name for name in shared_names
                             if self_names[name] != other_names[name]}
        if conflicting_names:
            raise CircuitError('Name conflict applying operation for parameters: '
                               '{}'.format(conflicting_names))

    def _apply_operation(self, operation, other, reflected=False):
        """Base method implementing math operations between Parameters and
        either a constant or a second ParameterExpression.

        Args:
            operation (function): One of operator.{add,sub,mul,truediv}.
            other (Parameter or numbers.Real): The second argument to be used
               with self in operation.
            reflected (bool): Optional - The default ordering is
                "self operator other". If reflected is True, this is switched
                to "other operator self". For use in e.g. __radd__, ...

        Raises:
            CircuitError:
                - If parameter_map contains Parameters outside those in self.
                - If the replacement Parameters in parameter_map would result in
                  a name conflict in the generated expression.

        Returns:
            ParameterExpression: a new expression describing the result of the
                operation.
        """

        self_expr = self._symbol_expr

        if isinstance(other, ParameterExpression):
            self._raise_if_parameter_names_conflict(other._parameter_symbols.keys())

            parameter_symbols = {**self._parameter_symbols, **other._parameter_symbols}
            other_expr = other._symbol_expr
        elif isinstance(other, numbers.Real) and numpy.isfinite(other):
            parameter_symbols = self._parameter_symbols.copy()
            other_expr = other
        else:
            return NotImplemented

        if reflected:
            expr = operation(other_expr, self_expr)
        else:
            expr = operation(self_expr, other_expr)

        return ParameterExpression(parameter_symbols, expr)

    def __add__(self, other):
        return self._apply_operation(operator.add, other)

    def __radd__(self, other):
        return self._apply_operation(operator.add, other, reflected=True)

    def __sub__(self, other):
        return self._apply_operation(operator.sub, other)

    def __rsub__(self, other):
        return self._apply_operation(operator.sub, other, reflected=True)

    def __mul__(self, other):
        return self._apply_operation(operator.mul, other)

    def __neg__(self):
        return self._apply_operation(operator.mul, -1.0)

    def __rmul__(self, other):
        return self._apply_operation(operator.mul, other, reflected=True)

    def __truediv__(self, other):
        if other == 0:
            raise ZeroDivisionError('Division of a ParameterExpression by zero.')
        return self._apply_operation(operator.truediv, other)

    def __rtruediv__(self, other):
        return self._apply_operation(operator.truediv, other, reflected=True)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, str(self))

    def __str__(self):
        return str(self._symbol_expr)

    def __float__(self):
        if self.parameters:
            raise TypeError('ParameterExpression with unbound parameters ({}) '
                            'cannot be cast to a float.'.format(self.parameters))
        return float(self._symbol_expr)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo=None):
        return self
