# -*- coding: utf-8 -*-

"""
adding extension function to Gate
"""
from qiskit.extensions.standard import XGate, ZGate, SwapGate, CnotGate
from z3 import Not, And

## HGate
XGate.postcondition = lambda x1,y1: y1
 
## XGate
XGate.postcondition = lambda x1,y1: y1==Not(x1)

## YGate  
YGate.postcondition = lambda x1,y1: y1=Not(x1)

## ZGate
ZGate.trivial_if = lambda x1: 1 
ZGate.postcondition = lambda x1,y1: y1==x1

##SwapGate 
SwapGate.trivial_if = lambda x1,x2: x1==x2
SwapGate.postcondition = lambda x1,x2,y1,y2: And(x1==y2, x2==y1)

