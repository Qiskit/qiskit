# -*- coding: utf-8 -*-

"""
Dynamically extend Gate classes with functions required for the Hoare
optimizer, namely triviality- and post-conditionsto.
A return value of 'true' for triviality conditions indicates the gate is
always trivial, provided the qubit is in a classical state.
Functions/gates that are omitted here are assumed to always be
non-trivial and/or have unknown post-conditions.
"""
from qiskit.extensions.standard import IdGate, XGate, YGate, ZGate
from qiskit.extensions.standard import CnotGate, ToffoliGate, CyGate, CzGate
from qiskit.extensions.standard import TGate, SGate, RZGate, U1Gate
from qiskit.extensions.standard import SwapGate, FredkinGate, CrzGate, Cu1Gate
from z3 import Not, And

# FLIP GATES #
# XGate
XGate.postconditions = lambda self, x1, y1: y1 == Not(x1)
CnotGate.postconditions = lambda self, x1, y1: y1 == Not(x1)
ToffoliGate.postconditions = lambda self, x1, y1: y1 == Not(x1)

# YGate
YGate.postconditions = lambda self, x1, y1: y1 == Not(x1)
CyGate.postconditions = lambda self, x1, y1: y1 == Not(x1)

# PHASE GATES #
# IdGate
IdGate.postconditions = lambda self, x1, y1: y1 == x1

# ZGate
ZGate.trivial_if = lambda self, x1: True
ZGate.postconditions = lambda self, x1, y1: y1 == x1
CzGate.trivial_if = lambda self, x1: True
CzGate.postconditions = lambda self, x1, y1: y1 == x1

# SGate
SGate.trivial_if = lambda self, x1: True
SGate.postconditions = lambda self, x1, y1: y1 == x1

# TGate
TGate.trivial_if = lambda self, x1: True
TGate.postconditions = lambda self, x1, y1: y1 == x1

# RzGate = U1Gate
RZGate.trivial_if = lambda self, x1: True
RZGate.postconditions = lambda self, x1, y1: y1 == x1
CrzGate.trivial_if = lambda self, x1: True
CrzGate.postconditions = lambda self, x1, y1: y1 == x1
U1Gate.trivial_if = lambda self, x1: True
U1Gate.postconditions = lambda self, x1, y1: y1 == x1
Cu1Gate.trivial_if = lambda self, x1: True
Cu1Gate.postconditions = lambda self, x1, y1: y1 == x1

# MULTI-QUBIT GATES #
# SwapGate
SwapGate.trivial_if = lambda self, x1, x2: x1 == x2
SwapGate.postconditions = lambda self, x1, x2, y1, y2: And(x1 == y2, x2 == y1)
FredkinGate.trivial_if = lambda self, x1, x2: x1 == x2
FredkinGate.postconditions = lambda self, x1, x2, y1, y2: And(x1 == y2, x2 == y1)
