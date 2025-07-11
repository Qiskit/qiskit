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

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Union
from uuid import uuid4, UUID
import numbers
import operator

import numpy as np

from qiskit.utils.optionals import HAS_SYMPY
from qiskit.circuit.exceptions import CircuitError
import qiskit._accelerate.circuit

ParameterExpressionBase = qiskit._accelerate.circuit.ParameterExpression
OPReplay = qiskit._accelerate.circuit.OPReplay


# This type is redefined at the bottom to insert the full reference to "ParameterExpression", so it
# can safely be used by runtime type-checkers like Sphinx.  Mypy does not need this because it
# handles the references by static analysis.
ParameterValueType = Union["ParameterExpression", float]


class _OPCode(IntEnum):
    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3
    POW = 4
    SIN = 5
    COS = 6
    TAN = 7
    ASIN = 8
    ACOS = 9
    EXP = 10
    LOG = 11
    SIGN = 12
    GRAD = 13
    CONJ = 14
    SUBSTITUTE = 15
    ABS = 16
    ATAN = 17
    RSUB = 18
    RDIV = 19
    RPOW = 20


_OP_CODE_MAP = (
    "__add__",
    "__sub__",
    "__mul__",
    "__truediv__",
    "__pow__",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "exp",
    "log",
    "sign",
    "gradient",
    "conjugate",
    "subs",
    "abs",
    "arctan",
    "__rsub__",
    "__rtruediv__",
    "__rpow__",
)


def op_code_to_method(op_code: _OPCode):
    """Return the method name for a given op_code."""
    return _OP_CODE_MAP[op_code]


def inst_to_parameter_class(expr):
    """Return Python Parameter/ParameterExpression from Rust ParameterExpression"""
    if isinstance(expr, ParameterExpressionBase):
        if expr.is_symbol:
            from .parameter import Parameter

            return Parameter(str(expr), UUID(int=expr.get_uuid()))
        else:
            return ParameterExpression(None, expr)
    else:
        # return value as is
        return expr


class ParameterExpression(ParameterExpressionBase):
    """ParameterExpression class to enable creating expressions of Parameters."""

    __slots__ = [
        "_parameter_symbols",
    ]

    def __new__(cls, symbol_map=None, expr=None):
        """Create a new :class:`ParameterExpression`.

        Not intended to be called directly, but to be instantiated via operations
        on other :class:`Parameter` or :class:`ParameterExpression` objects.
        The constructor of this object is **not** a public interface and should not
        ever be used directly.

        Args:
            symbol_map (Dict[Parameter, [ParameterExpression, float, or int]]):
                Mapping of :class:`Parameter` instances to the :class:`sympy.Symbol`
                serving as their placeholder in expr.
            expr (SymbolExpr or str): Expression with Rust's SymbolExprPy or string
        """
        # NOTE: `Parameter.__init__` does not call up to this method, since this method is dependent
        # on `Parameter` instances already being initialized enough to be hashable.  If changing
        # this method, check that `Parameter.__init__` and `__setstate__` are still valid.

        if symbol_map is not None:
            if isinstance(symbol_map, dict):
                symbol_map = set(symbol_map.keys())
        self = super().__new__(cls, symbol_map, expr)

        if symbol_map is not None:
            self._parameter_symbols = symbol_map
        else:
            self._parameter_symbols = set()
        return self

    @property
    def parameters(self) -> set:
        """Returns a set of the unbound Parameters in the expression."""
        return self._parameter_symbols

    @property
    def parameter_symbols_dict(self) -> dict:
        return dict(zip(self._parameter_symbols, self._parameter_symbols))

    @property
    def _qpy_replay(self) -> list:
        replay = self.replay()
        if replay is None:
            replay = []
        return replay

    def conjugate(self) -> "ParameterExpression":
        """Return the conjugate."""
        return ParameterExpression(self._parameter_symbols, super().py_conjugate())

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

    def bind(
        self, parameter_values: dict, allow_unknown_parameters: bool = False
    ) -> "ParameterExpression":
        """Binds the provided set of parameters to their corresponding values.

        Args:
            parameter_values: Mapping of Parameter instances to the numeric value to which
                              they will be bound.
            allow_unknown_parameters: If ``False``, raises an error if ``parameter_values``
                contains Parameters in the keys outside those present in the expression.
                If ``True``, any such parameters are simply ignored.

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
        parameters = self._parameter_symbols - parameter_values.keys()
        return ParameterExpression(
            parameters, super().py_bind(parameter_values, allow_unknown_parameters)
        )

    def subs(
        self, parameter_map: dict, allow_unknown_parameters: bool = False
    ) -> "ParameterExpression":
        """Returns a new Expression with replacement Parameters.

        Args:
            parameter_map: Mapping from Parameters in self to the ParameterExpression
                           instances with which they should be replaced.
            allow_unknown_parameters: If ``False``, raises an error if ``parameter_map``
                contains Parameters in the keys outside those present in the expression.
                If ``True``, any such parameters are simply ignored.

        Raises:
            CircuitError:
                - If parameter_map contains Parameters outside those in self.
                - If the replacement Parameters in parameter_map would result in
                  a name conflict in the generated expression.

        Returns:
            A new expression with the specified parameters replaced.
        """
        parameters = self._parameter_symbols - parameter_map.keys()
        for old_param, new_param in parameter_map.items():
            parameters = parameters | new_param._parameter_symbols

        return ParameterExpression(
            parameters, super().py_subs(parameter_map, allow_unknown_parameters)
        )

    def gradient(self, param) -> Union["ParameterExpression", complex]:
        """Get the derivative of a real parameter expression w.r.t. a specified parameter.

        .. note::

            This method assumes that the parameter expression represents a **real expression only**.
            Calling this method on a parameter expression that contains complex values, or binding
            complex values to parameters in the expression is undefined behavior.

        Args:
            param (Parameter): Parameter w.r.t. which we want to take the derivative

        Returns:
            ParameterExpression representing the gradient of param_expr w.r.t. param
            or complex or float number
        """
        expr_grad = super().py_gradient(param)
        if isinstance(expr_grad, ParameterExpressionBase):
            parameters = set()
            params = expr_grad.parameters
            for p in params:
                for q in self._parameter_symbols:
                    if p == str(q):
                        parameters.add(q)
            return ParameterExpression(parameters, expr_grad)
        return expr_grad

    def _merge_parameters(
        self,
        other: ParameterValueType,
    ) -> set:
        if isinstance(other, ParameterExpression):
            return self._parameter_symbols | other._parameter_symbols
        else:
            return self._parameter_symbols.copy()

    def __add__(self, other):
        return ParameterExpression(self._merge_parameters(other), super().py_add(other))

    def __radd__(self, other):
        return ParameterExpression(self._merge_parameters(other), super().py_radd(other))

    def __sub__(self, other):
        return ParameterExpression(self._merge_parameters(other), super().py_sub(other))

    def __rsub__(self, other):
        return ParameterExpression(self._merge_parameters(other), super().py_rsub(other))

    def __mul__(self, other):
        if isinstance(other, np.ndarray):
            return other * self
        return ParameterExpression(self._merge_parameters(other), super().py_mul(other))

    def __pos__(self):
        return ParameterExpression(self._parameter_symbols, super().py_pos())

    def __neg__(self):
        return ParameterExpression(self._parameter_symbols, super().py_neg())

    def __rmul__(self, other):
        if isinstance(other, np.ndarray):
            return other * self
        return ParameterExpression(self._merge_parameters(other), super().py_rmul(other))

    def __truediv__(self, other):
        if other == 0:
            raise ZeroDivisionError("Division of a ParameterExpression by zero.")
        return ParameterExpression(self._merge_parameters(other), super().py_div(other))

    def __rtruediv__(self, other):
        return ParameterExpression(self._merge_parameters(other), super().py_rdiv(other))

    def __pow__(self, other):
        return ParameterExpression(self._merge_parameters(other), super().py_pow(other))

    def __rpow__(self, other):
        return ParameterExpression(self._merge_parameters(other), super().py_rpow(other))

    def sin(self):
        """Sine of a ParameterExpression"""
        return ParameterExpression(self._parameter_symbols, super().py_sin())

    def cos(self):
        """Cosine of a ParameterExpression"""
        return ParameterExpression(self._parameter_symbols, super().py_cos())

    def tan(self):
        """Tangent of a ParameterExpression"""
        return ParameterExpression(self._parameter_symbols, super().py_tan())

    def arcsin(self):
        """Arcsin of a ParameterExpression"""
        return ParameterExpression(self._parameter_symbols, super().py_arcsin())

    def arccos(self):
        """Arccos of a ParameterExpression"""
        return ParameterExpression(self._parameter_symbols, super().py_arccos())

    def arctan(self):
        """Arctan of a ParameterExpression"""
        return ParameterExpression(self._parameter_symbols, super().py_arctan())

    def exp(self):
        """Exponential of a ParameterExpression"""
        return ParameterExpression(self._parameter_symbols, super().py_exp())

    def log(self):
        """Logarithm of a ParameterExpression"""
        return ParameterExpression(self._parameter_symbols, super().py_log())

    def sign(self):
        """Sign of a ParameterExpression"""
        return ParameterExpression(self._parameter_symbols, super().py_sign())

    def __abs__(self):
        """Absolute of a ParameterExpression"""
        return ParameterExpression(self._parameter_symbols, super().py_abs())

    def abs(self):
        """Absolute of a ParameterExpression"""
        return self.__abs__()

    def __repr__(self):
        return f"{self.__class__.__name__}({str(self)})"

    def __copy__(self):
        return self

    def __deepcopy__(self, memo=None):
        return self

    @HAS_SYMPY.require_in_call
    def sympify(self):
        """Return symbolic expression as a raw Sympy object.

        .. note::

            This is for interoperability only.  Qiskit will not accept or work with raw Sympy or
            Symegine expressions in its parameters, because they do not contain the tracking
            information used in circuit-parameter binding and assignment.
        """
        import sympy

        if self.replay() is None:
            if self.is_symbol:
                return sympy.Symbol(super().sympify())
            else:
                return sympy.sympify(super().sympify())

        output = None
        for inst in self.replay():
            if isinstance(inst, OPReplay._SUBS):
                sympy_binds = {}
                for old, new in inst.binds.items():
                    if isinstance(new, ParameterExpressionBase):
                        new = new.sympify()
                    sympy_binds[old.sympify()] = new
                output = output.subs(sympy_binds, simultaneous=True)
                continue

            if isinstance(inst.lhs, ParameterExpressionBase):
                if inst.lhs.replay() is None:
                    if inst.lhs.is_symbol:
                        lhs = sympy.Symbol(inst.lhs.sympify())
                    else:
                        lhs = sympy.sympify(inst.lhs.sympify())
                else:
                    lhs = ParameterExpression(None, inst.lhs).sympify()
            elif inst.lhs is None:
                lhs = output
            else:
                lhs = inst.lhs

            method_str = _OP_CODE_MAP[int(inst.op)]
            if int(inst.op) in {0, 1, 2, 3, 4, 13, 15, 18, 19, 20}:
                if inst.rhs is None:
                    rhs = output
                elif isinstance(inst.rhs, ParameterExpressionBase):
                    if inst.rhs.replay() is None:
                        if inst.rhs.is_symbol:
                            rhs = sympy.Symbol(inst.rhs.sympify())
                        else:
                            rhs = sympy.sympify(inst.rhs.sympify())
                    else:
                        rhs = ParameterExpression(None, inst.rhs).sympify()
                else:
                    rhs = inst.rhs

                if (
                    not isinstance(lhs, sympy.Basic)
                    and isinstance(rhs, sympy.Basic)
                    and int(inst.op) in [0, 2]
                ):
                    if int(inst.op) == 0:
                        method_str = "__radd__"
                    elif int(inst.op) == 2:
                        method_str = "__rmul__"
                    output = getattr(rhs, method_str)(lhs)
                elif int(inst.op) == _OPCode.GRAD:
                    output = getattr(lhs, "diff")(rhs)
                else:
                    output = getattr(lhs, method_str)(rhs)
            else:
                if int(inst.op) == _OPCode.ACOS:
                    output = getattr(sympy, "acos")(lhs)
                elif int(inst.op) == _OPCode.ASIN:
                    output = getattr(sympy, "asin")(lhs)
                elif int(inst.op) == _OPCode.ATAN:
                    output = getattr(sympy, "atan")(lhs)
                elif int(inst.op) == _OPCode.ABS:
                    output = getattr(sympy, "Abs")(lhs)
                else:
                    output = getattr(sympy, method_str)(lhs)
        return output


# Redefine the type so external imports get an evaluated reference; Sphinx needs this to understand
# the type hints.
ParameterValueType = Union[ParameterExpression, float]
