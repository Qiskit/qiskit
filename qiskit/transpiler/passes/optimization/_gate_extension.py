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
Dynamically extend Gate classes with functions required for the Hoare
optimizer, namely triviality- and post-conditionsto.
A return value of 'true' for triviality conditions indicates the gate is
always trivial, provided the qubit is in a classical state.
Functions/gates that are omitted here are assumed to always be
non-trivial and/or have unknown post-conditions.
"""
from z3 import Not, And
from qiskit.extensions.standard import IdGate, XGate, YGate, ZGate
from qiskit.extensions.standard import CnotGate, ToffoliGate, CyGate, CzGate
from qiskit.extensions.standard import TGate, SGate, RZGate, U1Gate
from qiskit.extensions.standard import SwapGate, FredkinGate, CrzGate, Cu1Gate

# FLIP GATES #
# XGate
XGate._postconditions = lambda self, x1, y1: y1 == Not(x1)
CnotGate._postconditions = lambda self, x1, y1: y1 == Not(x1)
ToffoliGate._postconditions = lambda self, x1, y1: y1 == Not(x1)

# YGate
YGate._postconditions = lambda self, x1, y1: y1 == Not(x1)
CyGate._postconditions = lambda self, x1, y1: y1 == Not(x1)

# PHASE GATES #
# IdGate
IdGate._postconditions = lambda self, x1, y1: y1 == x1

# ZGate
ZGate._trivial_if = lambda self, x1: True
ZGate._postconditions = lambda self, x1, y1: y1 == x1
CzGate._trivial_if = lambda self, x1: True
CzGate._postconditions = lambda self, x1, y1: y1 == x1

# SGate
SGate._trivial_if = lambda self, x1: True
SGate._postconditions = lambda self, x1, y1: y1 == x1

# TGate
TGate._trivial_if = lambda self, x1: True
TGate._postconditions = lambda self, x1, y1: y1 == x1

# RzGate = U1Gate
RZGate._trivial_if = lambda self, x1: True
RZGate._postconditions = lambda self, x1, y1: y1 == x1
CrzGate._trivial_if = lambda self, x1: True
CrzGate._postconditions = lambda self, x1, y1: y1 == x1
U1Gate._trivial_if = lambda self, x1: True
U1Gate._postconditions = lambda self, x1, y1: y1 == x1
Cu1Gate._trivial_if = lambda self, x1: True
Cu1Gate._postconditions = lambda self, x1, y1: y1 == x1

# MULTI-QUBIT GATES #
# SwapGate
SwapGate._trivial_if = lambda self, x1, x2: x1 == x2
SwapGate._postconditions = lambda self, x1, x2, y1, y2: And(x1 == y2, x2 == y1)
FredkinGate._trivial_if = lambda self, x1, x2: x1 == x2
FredkinGate._postconditions = lambda self, x1, x2, y1, y2: And(x1 == y2, x2 == y1)
