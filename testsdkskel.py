"""
Test the SDK skeleton.

Author: Andrew Cross
"""
import math
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.extensions.standard import h, cx, u1, u2, u3, iden, x, y, z, s
from qiskit.extensions.standard import t, ccx, cswap

# Issues
#   .q_if is not implemented, store controls on each gate, two methods

# Ismael advice - keep it simple

# how do add registers as I go?

# q = qr("new",5)
# p.add(q)
# use q

# q.barrier()
# q.h(0)
# q.h(3)
# q.barrier()

# q.h(0)
# q.id(0)
# q.h(0)

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
#    q.cx(i, i)  # this raises exception
for i in range(n):
    p.u1(math.pi / (i+1), q[i])
    p.h(q[i])
    p.measure(q[i], c[i])
q.ccx(0, 1, 2)
print(p.qasm())
print(pp.qasm())

pq1 = QuantumRegister("q", n)
pc1 = ClassicalRegister("c", n)
p1 = QuantumCircuit(pq1, pc1)
pq2 = QuantumRegister("q", n)
# pq2 = QuantumRegister("qq", n)  # this raises exception
# pq2 = QuantumRegister("q", n-1)  # this raises exception
p2 = QuantumCircuit(pq2)
pq2.barrier()
pq1.s(0).inverse()
pq2.s(1)
pq2.cswap(0, 1, 2)
p1.measure(pq1[0], pc1[0]).c_if(pc1, 3)
p1 += p2
p3 = p1 + p2
print(p1.qasm())
print(p3.qasm())
