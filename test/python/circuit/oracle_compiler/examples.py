# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name, missing-function-docstring

"""These examples should be handle by the oracle compiler"""

from qiskit.circuit.oracle_compiler.types import Bit


def identity(a: Bit) -> Bit:
    return a


def bit_and(a: Bit, b: Bit) -> Bit:
    return a & b


def bit_or(a: Bit, b: Bit) -> Bit:
    return a | b


def bool_or(a: Bit, b: Bit) -> Bit:
    return a or b


def bool_not(a: Bit) -> Bit:
    return not a


def and_and(a: Bit, b: Bit, c: Bit) -> Bit:
    return a and b and c


def multiple_binop(a: Bit, b: Bit) -> Bit:
    return (a or b) | (b & a) and (a & b)


def id_assing(a: Bit) -> Bit:
    b = a
    return b


def example1(a: Bit, b: Bit) -> Bit:
    c = a & b
    d = b | a
    return c ^ a | d


def grover_oracle(a: Bit, b: Bit, c: Bit, d: Bit) -> Bit:
    return not a and b and not c and d
