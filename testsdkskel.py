"""
Test the SDK skeleton.

Author: Andrew Cross
"""
import math
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.extensions.standard import h, cx, u1, u2, u3, iden, x, y, z, s
from qiskit.extensions.standard import t, ccx

# Issues
#   .q_if is not implemented

# Ismael - keep it simple

n = 5
q = QuantumRegister("q", n)
r = QuantumRegister("r", 2*n)
c = ClassicalRegister("c", n)
cc = ClassicalRegister("cc", n)
# q.h(0)  # this raises exception
p = QuantumCircuit(q, c)
# pp = QuantumCircuit(q, r, c) # this raises exception
pp = QuantumCircuit(r, cc)
q.reset()  # this applies the reset gate to q in p
q.h(0)
q.h(1).c_if(c, 5)
# r.h(2).c_if(c, 5)  # raise exception
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
for i in range(n):
    p.u1(math.pi / (i+1), q[i])
    p.h(q[i])
    p.measure(q[i], c[i])
q.ccx(0, 1, 2)
print(p.qasm())
print(pp.qasm())
