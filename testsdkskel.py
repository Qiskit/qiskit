"""
Test the SDK skeleton.

Author: Andrew Cross
"""
import math
from qiskit_sdk import QuantumRegister, ClassicalRegister, Program
from qiskit_sdk.extensions.standard import h, cx, u1, u2, u3, iden, x, y, z, s
from qiskit_sdk.extensions.standard import t, ccx

# For the standard library, we have chosen to unroll as we go.
# This is the simplest solution and should not lead to any performance
# issues in the near future. We can modify the implementation later to
# postpone expansion of the composite gates. We can also compress the
# QASM before transmitting it through the web API.

# Issues
#   .q_if is not implemented

# What other features do we need?
# - Think about how we run programs and get results.
# - Think about return values from measurements.

n = 5
q = QuantumRegister("q", n)
r = QuantumRegister("r", 2*n)
c = ClassicalRegister("c", n)
# q.h(0)  # this raises exception
p = Program(q, c)
pp = Program(q, r, c)  # it would be easy to bail here if that's preferred
q.reset()
q.h(0)  # this gets applied to q in p and pp
q.h(1).c_if(c, 5)
ppp = Program(q)
#q.h(2).c_if(c, 5)  # raise exception
q.u1(math.pi/4.0).inverse().inverse()
q.u2(math.pi/8.0, math.pi/8.0)
q.u3(math.pi/4.0, math.pi/8.0, math.pi/16.0)
r.iden(0)
r.s().inverse()
r.barrier()
r.x()
r.y()
r.z()
r.t()
r.reset(0)
for i in range(n-1):
    q.cx(i, i+1)
thetarray = [math.pi/x for x in range(1, n+1)]
for i in range(n):
    p.u1(thetarray[i], (q, i))
    p.h((q, i))
    p.measure((q, i), (c, i))
    pp.measure((q, i), (c, i))
q.ccx(0, 1, 2)
print(p.qasm())
print(pp.qasm())
