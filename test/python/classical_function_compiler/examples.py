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

# pylint: disable=missing-function-docstring

"""These examples should be handle by the classicalfunction compiler"""

from qiskit.circuit import Int1


def identity(a: Int1) -> Int1:
    return a


def bit_and(a: Int1, b: Int1) -> Int1:
    return a & b


def bit_or(a: Int1, b: Int1) -> Int1:
    return a | b


def bool_or(a: Int1, b: Int1) -> Int1:
    return a or b


def bool_not(a: Int1) -> Int1:
    return not a


def and_and(a: Int1, b: Int1, c: Int1) -> Int1:
    return a and b and c


def multiple_binop(a: Int1, b: Int1) -> Int1:
    return (a or b) | (b & a) and (a & b)


def id_assing(a: Int1) -> Int1:
    b = a
    return b


def example1(a: Int1, b: Int1) -> Int1:
    c = a & b
    d = b | a
    return c ^ a | d


def grover_oracle(a: Int1, b: Int1, c: Int1, d: Int1) -> Int1:
    return not a and b and not c and d
