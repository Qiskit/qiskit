# -*- coding: utf-8 -*-

"""
adding extension function to Gate
"""
from qiskit.extensions.standard import HGate, XGate, YGate, ZGate, SwapGate, CnotGate, ToffoliGate
from z3 import Not, And

# HGate
HGate.postconditions = lambda self, x1, y1: True

# XGate
XGate.postconditions = lambda self, x1, y1: y1 == Not(x1)

# YGate
YGate.postconditions = lambda self, x1, y1: y1 == Not(x1)

# ZGate
ZGate.trivial_if = lambda self, x1: True
ZGate.postconditions = lambda self, x1, y1: y1 == x1

# CnotGate
CnotGate.postconditions = lambda self, x1, y1: y1 == Not(x1)

# ToffoliGate
ToffoliGate.postconditions = lambda self, x1, y1: y1 == Not(x1)

# SwapGate
SwapGate.trivial_if = lambda self, x1, x2: x1 == x2
SwapGate.postconditions = lambda self, x1, x2, y1, y2: And(x1 == y2, x2 == y1)
