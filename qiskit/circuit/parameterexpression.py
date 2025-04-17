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
from typing import Union

import qiskit._accelerate.circuit

ParameterExpression = qiskit._accelerate.circuit.ParameterExpression


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


ParameterValueType = Union[ParameterExpression, float]


@dataclass
class _INSTRUCTION:
    op: _OPCode
    lhs: ParameterValueType | None
    rhs: ParameterValueType | None = None


@dataclass
class _SUBS:
    binds: dict
    op: _OPCode = _OPCode.SUBSTITUTE
