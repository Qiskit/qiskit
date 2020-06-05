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

"""
Operator Globals
"""

from qiskit.quantum_info import Pauli
from qiskit.circuit.library import CXGate, SGate, TGate, HGate, SwapGate, CZGate

from .primitive_ops.primitive_op import PrimitiveOp
from .state_fns.state_fn import StateFn

# pylint: disable=invalid-name

# Digits of precision when returning values from eval functions. Without rounding, 1e-17 or 1e-32
# values often show up in place of 0, etc.
# Note: care needs to be taken in rounding otherwise some behavior may not be as expected. E.g
# evolution is used in QAOA variational form and difference when optimizing may be small - round
# the outcome too much and a small difference may become none and the optimizer gets stuck where
# otherwise it would not.
EVAL_SIG_DIGITS = 18

# Immutable convenience objects


def make_immutable(obj):
    """ Delete the __setattr__ property to make the object mostly immutable. """

    # TODO figure out how to get correct error message
    # def throw_immutability_exception(self, *args):
    #     raise AquaError('Operator convenience globals are immutable.')

    obj.__setattr__ = None
    return obj


# 1-Qubit Paulis
X = make_immutable(PrimitiveOp(Pauli.from_label('X')))
Y = make_immutable(PrimitiveOp(Pauli.from_label('Y')))
Z = make_immutable(PrimitiveOp(Pauli.from_label('Z')))
I = make_immutable(PrimitiveOp(Pauli.from_label('I')))

# Clifford+T, and some other common non-parameterized gates
CX = make_immutable(PrimitiveOp(CXGate()))
S = make_immutable(PrimitiveOp(SGate()))
H = make_immutable(PrimitiveOp(HGate()))
T = make_immutable(PrimitiveOp(TGate()))
Swap = make_immutable(PrimitiveOp(SwapGate()))
CZ = make_immutable(PrimitiveOp(CZGate()))

# 1-Qubit Paulis
Zero = make_immutable(StateFn('0'))
One = make_immutable(StateFn('1'))
Plus = make_immutable(H.compose(Zero))
Minus = make_immutable(H.compose(X).compose(Zero))
