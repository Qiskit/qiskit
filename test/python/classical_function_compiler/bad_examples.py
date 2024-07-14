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

# pylint: disable=missing-function-docstring, undefined-variable

"""These are bad examples and raise errors in in the classicalfunction compiler"""

from qiskit.circuit.classicalfunction.types import Int1, Int2


def id_no_type_arg(a) -> Int1:
    return a


def id_no_type_return(a: Int1):
    return a


def id_bad_return(a: Int1) -> Int2:
    return a


def out_of_scope(a: Int1) -> Int1:
    return a & c


def bit_not(a: Int1) -> Int1:
    # Bitwise not does not operate on booleans (aka, bits), but int
    return ~a
