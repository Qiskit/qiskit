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
from typing import Callable, Dict, Set, Union

import numbers
import operator

import numpy

from qiskit.circuit.exceptions import CircuitError
from qiskit.utils import optionals as _optionals

# This type is redefined at the bottom to insert the full reference to "ParameterExpression", so it
# can safely be used by runtime type-checkers like Sphinx.  Mypy does not need this because it
# handles the references by static analysis.
ParameterValueType = Union["ParameterExpression", float]


class ParameterExpression:
    """ParameterExpression class to enable creating expressions of Parameters."""

    __slots__ = ["_parameter_symbols", "_parameters", "_symbol_expr", "_name_map"]

    def __init__(self, symbol_map: Dict, expr):
        """Create a new :class:`ParameterExpression`.

        Not intended to be called directly, but to be instantiated via operations
        on other :class:`Parameter` or :class:`ParameterExpression` objects.

        Args:
            symbol_map (Dict[Parameter, [ParameterExpression, float, or int]]):
                Mapping of :class:`Parameter` instances to the :class:`sympy.Symbol`
                serving as their placeholder in expr.
            expr (sympy.Expr): Expression of :class:`sympy.Symbol` s.
        """
        self._parameter_symbols = symbol_map
        self._parameters = set(self._parameter_symbols)
        self._symbol_expr = expr
        self._name_map = None

    @property
    def parameters(self) -> Set:
        """Returns a set of the unbound Parameters in the expression."""
        return self._parameters

    @property
    def _names(self) -> Dict:
        """Returns a mapping of parameter names to Parameters in the expression."""
        if self._name_map is None:
            self._name_map = {p.name: p for p in self._parameters}
        return self._name_map

    def conjugate(self) -> "ParameterExpression":
        """Return the conjugate."""
        if _optionals.HAS_SYMENGINE:
            import symengine

            conjugated = ParameterExpression(
                self._parameter_symbols, symengine.conjugate(self._symbol_expr)
            )
        else:
            conjugated = ParameterExpression(self._parameter_symbols, self._symbol_expr.conjugate())
        return conjugated

    def assign(self, parameter, value: ParameterValueType) -> "ParameterExpression":
        """
        Assign one parameter to a value, which can either be numeric or another parameter
        expression.

        Args:
            parameter (Parameter): A parameter in this expression whose value will be updated.
            value: The new value to bind to.

        Returns:
            A new expression parameterized by any parameters which were not bound by assignment.
        """
        if isinstance(value, ParameterExpression):
            return self.subs({parameter: value})
        return self.bind({parameter: value})

    def bind(self, parameter_values: Dict) -> "ParameterExpression":
        """Binds the provided set of parameters to their corresponding values.

        Args:
            parameter_values: Mapping of Parameter instances to the numeric value to which
                              they will be bound.

        Raises:
            CircuitError:
                - If parameter_values contains Parameters outside those in self.
                - If a non-numeric value is passed in parameter_values.
            ZeroDivisionError:
                - If binding the provided values requires division by zero.

        Returns:
            A new expression parameterized by any parameters which were not bound by
            parameter_values.
        """

        self._raise_if_passed_unknown_parameters(parameter_values.keys())
        self._raise_if_passed_nan(parameter_values)

        symbol_values = {}
        for parameter, value in parameter_values.items():
            param_expr = self._parameter_symbols[parameter]
            symbol_values[param_expr] = value

        bound_symbol_expr = self._symbol_expr.subs(symbol_values)

        # Don't use sympy.free_symbols to count remaining parameters here.
        # sympy will in some cases reduce the expression and remove even
        # unbound symbols.
        # e.g. (sympy.Symbol('s') * 0).free_symbols == set()

        free_parameters = self.parameters - parameter_values.keys()
        free_parameter_symbols = {
            p: s for p, s in self._parameter_symbols.items() if p in free_parameters
        }

        if (
            hasattr(bound_symbol_expr, "is_infinite") and bound_symbol_expr.is_infinite
        ) or bound_symbol_expr == float("inf"):
            raise ZeroDivisionError(
                "Binding provided for expression "
                "results in division by zero "
                "(Expression: {}, Bindings: {}).".format(self, parameter_values)
            )

        return ParameterExpression(free_parameter_symbols, bound_symbol_expr)

    def subs(self, parameter_map: Dict) -> "ParameterExpression":
        """Returns a new Expression with replacement Parameters.

        Args:
            parameter_map: Mapping from Parameters in self to the ParameterExpression
                           instances with which they should be replaced.

        Raises:
            CircuitError:
                - If parameter_map contains Parameters outside those in self.
                - If the replacement Parameters in parameter_map would result in
                  a name conflict in the generated expression.

        Returns:
            A new expression with the specified parameters replaced.
        """
        inbound_parameters = set()
        inbound_names = {}
        for replacement_expr in parameter_map.values():
            for p in replacement_expr.parameters:
                inbound_parameters.add(p)
                inbound_names[p.name] = p

        self._raise_if_passed_unknown_parameters(parameter_map.keys())
        self._raise_if_parameter_names_conflict(inbound_names, parameter_map.keys())
        if _optionals.HAS_SYMENGINE:
            import symengine

            new_parameter_symbols = {p: symengine.Symbol(p.name) for p in inbound_parameters}
        else:
            from sympy import Symbol

            new_parameter_symbols = {p: Symbol(p.name) for p in inbound_parameters}

        # Include existing parameters in self not set to be replaced.
        new_parameter_symbols.update(
            {p: s for p, s in self._parameter_symbols.items() if p not in parameter_map}
        )

        # If new_param is an expr, we'll need to construct a matching sympy expr
        # but with our sympy symbols instead of theirs.

        symbol_map = {
            self._parameter_symbols[old_param]: new_param._symbol_expr
            for old_param, new_param in parameter_map.items()
        }

        substituted_symbol_expr = self._symbol_expr.subs(symbol_map)

        return ParameterExpression(new_parameter_symbols, substituted_symbol_expr)

    def _raise_if_passed_unknown_parameters(self, parameters):
        unknown_parameters = parameters - self.parameters
        if unknown_parameters:
            raise CircuitError(
                "Cannot bind Parameters ({}) not present in "
                "expression.".format([str(p) for p in unknown_parameters])
            )

    def _raise_if_passed_nan(self, parameter_values):
        nan_parameter_values = {
            p: v for p, v in parameter_values.items() if not isinstance(v, numbers.Number)
        }
        if nan_parameter_values:
            raise CircuitError(
                f"Expression cannot bind non-numeric values ({nan_parameter_values})"
            )

    def _raise_if_parameter_names_conflict(self, inbound_parameters, outbound_parameters=None):
        if outbound_parameters is None:
            outbound_parameters = set()
            outbound_names = {}
        else:
            outbound_names = {p.name: p for p in outbound_parameters}

        inbound_names = inbound_parameters
        conflicting_names = []
        for name, param in inbound_names.items():
            if name in self._names and name not in outbound_names:
                if param != self._names[name]:
                    conflicting_names.append(name)
        if conflicting_names:
            raise CircuitError(
                f"Name conflict applying operation for parameters: {conflicting_names}"
            )

    def _apply_operation(
        self, operation: Callable, other: ParameterValueType, reflected: bool = False
    ) -> "ParameterExpression":
        """Base method implementing math operations between Parameters and
        either a constant or a second ParameterExpression.

        Args:
            operation: One of operator.{add,sub,mul,truediv}.
            other: The second argument to be used with self in operation.
            reflected: Optional - The default ordering is "self operator other".
                       If reflected is True, this is switched to "other operator self".
                       For use in e.g. __radd__, ...

        Raises:
            CircuitError:
                - If parameter_map contains Parameters outside those in self.
                - If the replacement Parameters in parameter_map would result in
                  a name conflict in the generated expression.

        Returns:
            A new expression describing the result of the operation.
        """
        self_expr = self._symbol_expr

        if isinstance(other, ParameterExpression):
            self._raise_if_parameter_names_conflict(other._names)
            parameter_symbols = {**self._parameter_symbols, **other._parameter_symbols}
            other_expr = other._symbol_expr
        elif isinstance(other, numbers.Number) and numpy.isfinite(other):
            parameter_symbols = self._parameter_symbols.copy()
            other_expr = other
        else:
            return NotImplemented

        if reflected:
            expr = operation(other_expr, self_expr)
        else:
            expr = operation(self_expr, other_expr)

        out_expr = ParameterExpression(parameter_symbols, expr)
        out_expr._name_map = self._names.copy()
        if isinstance(other, ParameterExpression):
            out_expr._names.update(other._names.copy())

        return out_expr

    def gradient(self, param) -> Union["ParameterExpression", complex]:
        """Get the derivative of a parameter expression w.r.t. a specified parameter expression.

        Args:
            param (Parameter): Parameter w.r.t. which we want to take the derivative

        Returns:
            ParameterExpression representing the gradient of param_expr w.r.t. param
            or complex or float number
        """
        # Check if the parameter is contained in the parameter expression
        if param not in self._parameter_symbols.keys():
            # If it is not contained then return 0
            return 0.0

        # Compute the gradient of the parameter expression w.r.t. param
        key = self._parameter_symbols[param]
        if _optionals.HAS_SYMENGINE:
            import symengine

            expr_grad = symengine.Derivative(self._symbol_expr, key)
        else:
            # TODO enable nth derivative
            from sympy import Derivative

            expr_grad = Derivative(self._symbol_expr, key).doit()

        # generate the new dictionary of symbols
        # this needs to be done since in the derivative some symbols might disappear (e.g.
        # when deriving linear expression)
        parameter_symbols = {}
        for parameter, symbol in self._parameter_symbols.items():
            if symbol in expr_grad.free_symbols:
                parameter_symbols[parameter] = symbol
        # If the gradient corresponds to a parameter expression then return the new expression.
        if len(parameter_symbols) > 0:
            return ParameterExpression(parameter_symbols, expr=expr_grad)
        # If no free symbols left, return a complex or float gradient
        expr_grad_cplx = complex(expr_grad)
        if expr_grad_cplx.imag != 0:
            return expr_grad_cplx
        else:
            return float(expr_grad)

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
            raise ZeroDivisionError("Division of a ParameterExpression by zero.")
        return self._apply_operation(operator.truediv, other)

    def __rtruediv__(self, other):
        return self._apply_operation(operator.truediv, other, reflected=True)

    def _call(self, ufunc):
        return ParameterExpression(self._parameter_symbols, ufunc(self._symbol_expr))

    def sin(self):
        """Sine of a ParameterExpression"""
        if _optionals.HAS_SYMENGINE:
            import symengine

            return self._call(symengine.sin)
        else:
            from sympy import sin as _sin

            return self._call(_sin)

    def cos(self):
        """Cosine of a ParameterExpression"""
        if _optionals.HAS_SYMENGINE:
            import symengine

            return self._call(symengine.cos)
        else:
            from sympy import cos as _cos

            return self._call(_cos)

    def tan(self):
        """Tangent of a ParameterExpression"""
        if _optionals.HAS_SYMENGINE:
            import symengine

            return self._call(symengine.tan)
        else:
            from sympy import tan as _tan

            return self._call(_tan)

    def arcsin(self):
        """Arcsin of a ParameterExpression"""
        if _optionals.HAS_SYMENGINE:
            import symengine

            return self._call(symengine.asin)
        else:
            from sympy import asin as _asin

            return self._call(_asin)

    def arccos(self):
        """Arccos of a ParameterExpression"""
        if _optionals.HAS_SYMENGINE:
            import symengine

            return self._call(symengine.acos)
        else:
            from sympy import acos as _acos

            return self._call(_acos)

    def arctan(self):
        """Arctan of a ParameterExpression"""
        if _optionals.HAS_SYMENGINE:
            import symengine

            return self._call(symengine.atan)
        else:
            from sympy import atan as _atan

            return self._call(_atan)

    def exp(self):
        """Exponential of a ParameterExpression"""
        if _optionals.HAS_SYMENGINE:
            import symengine

            return self._call(symengine.exp)
        else:
            from sympy import exp as _exp

            return self._call(_exp)

    def log(self):
        """Logarithm of a ParameterExpression"""
        if _optionals.HAS_SYMENGINE:
            import symengine

            return self._call(symengine.log)
        else:
            from sympy import log as _log

            return self._call(_log)

    def __repr__(self):
        return f"{self.__class__.__name__}({str(self)})"

    def __str__(self):
        from sympy import sympify, sstr

        return sstr(sympify(self._symbol_expr), full_prec=False)

    def __complex__(self):
        try:
            return complex(self._symbol_expr)
        # TypeError is for sympy, RuntimeError for symengine
        except (TypeError, RuntimeError) as exc:
            raise TypeError(
                "ParameterExpression with unbound parameters ({}) "
                "cannot be cast to a complex.".format(self.parameters)
            ) from exc

    def __float__(self):
        try:
            return float(self._symbol_expr)
        # TypeError is for sympy, RuntimeError for symengine
        except (TypeError, RuntimeError) as exc:
            raise TypeError(
                "ParameterExpression with unbound parameters ({}) "
                "cannot be cast to a float.".format(self.parameters)
            ) from exc

    def __int__(self):
        try:
            return int(self._symbol_expr)
        # TypeError is for sympy, RuntimeError for symengine
        except (TypeError, RuntimeError) as exc:
            raise TypeError(
                "ParameterExpression with unbound parameters ({}) "
                "cannot be cast to an int.".format(self.parameters)
            ) from exc

    def __hash__(self):
        return hash((frozenset(self._parameter_symbols), self._symbol_expr))

    def __copy__(self):
        return self

    def __deepcopy__(self, memo=None):
        return self

    def __eq__(self, other):
        """Check if this parameter expression is equal to another parameter expression
           or a fixed value (only if this is a bound expression).
        Args:
            other (ParameterExpression or a number):
                Parameter expression or numeric constant used for comparison
        Returns:
            bool: result of the comparison
        """
        if isinstance(other, ParameterExpression):
            if self.parameters != other.parameters:
                return False
            if _optionals.HAS_SYMENGINE:
                from sympy import sympify

                return sympify(self._symbol_expr).equals(sympify(other._symbol_expr))
            else:
                return self._symbol_expr.equals(other._symbol_expr)
        elif isinstance(other, numbers.Number):
            return len(self.parameters) == 0 and complex(self._symbol_expr) == other
        return False

    def is_real(self):
        """Return whether the expression is real"""

        if not self._symbol_expr.is_real and self._symbol_expr.is_real is not None:
            # Symengine returns false for is_real on the expression if
            # there is a imaginary component (even if that component is 0),
            # but the parameter will evaluate as real. Check that if the
            # expression's is_real attribute returns false that we have a
            # non-zero imaginary
            if _optionals.HAS_SYMENGINE:
                if self._symbol_expr.imag != 0.0:
                    return False
            else:
                return False
        return True

    def sympify(self):
        """Return symbolic expression as a raw Sympy or Symengine object.

        Symengine is used preferentially; if both are available, the result will always be a
        ``symengine`` object.  Symengine is a separate library but has integration with Sympy.

        .. note::

            This is for interoperability only.  Qiskit will not accept or work with raw Sympy or
            Symegine expressions in its parameters, because they do not contain the tracking
            information used in circuit-parameter binding and assignment.
        """
        return self._symbol_expr


# Redefine the type so external imports get an evaluated reference; Sphinx needs this to understand
# the type hints.
ParameterValueType = Union[ParameterExpression, float]
