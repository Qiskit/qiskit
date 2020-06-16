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

# pylint: disable=invalid-name, missing-function-docstring, undefined-variable

"""These are bad examples and raise errors in in the oracle compiler"""

from qiskit.transpiler.oracle_compiler.types import Bit, int2


def id_no_type_arg(a) -> Bit:
    return a


def id_no_type_return(a: Bit):
    return a


def id_bad_return(a: Bit) -> int2:
    return a


def out_of_scope(a: Bit) -> Bit:
    return a & c


def bit_not(a: Bit) -> Bit:
    # Bitwise not does not operate on booleans (aka, bits), but int
    return ~a
