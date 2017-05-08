"""
Test the SDK skeleton.

Author: Andrew Cross
"""
import sys
sys.path.append("..")
import math
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.extensions.standard import barrier, ubase, cxbase
from qiskit.extensions.standard import h, cx, u1, u2, u3, iden, x, y, z, s
from qiskit.extensions.standard import t, ccx, cswap

# Issues
#   .q_if is not implemented, store controls on each gate, two methods

# Ismael advice - keep it simple

n = 5
q = QuantumRegister("q", n)
r = QuantumRegister("r", 2*n)
c = ClassicalRegister("c", n)
cc = ClassicalRegister("cc", n)

qc = QuantumCircuit(q, c)
# can also do this:
# qc = QuantumCircuit()
# qc.add(q)
# qc.add(c)
# or this:
# qc = QuantumCircuit(QuantumRegister("q", n),
#                     ClassicalRegister("c", n))
# q = qc.regs["q"]
# c = qc.regs["c"]
qc2 = QuantumCircuit(q, r, c)  # this is OK, q just refers to any "qreg q[n];"
qc3 = QuantumCircuit(r, cc)

qc.reset(q)
qc.barrier()  # barrier on all qubits in qc
qc.h(q[0])
qc.h(q[1]).c_if(c, 5)

# qc3.h(r[2]).c_if(c, 5)  # raise exception

qc.u1(math.pi/4.0, q).inverse().inverse()
qc.u2(math.pi/8.0, math.pi/8.0, q)
qc.u3(math.pi/4.0, math.pi/8.0, math.pi/16.0, q)

qc3.iden(r[0])
qc3.s(r).inverse()
qc3.barrier(r)

qc3.x(r)
qc3.y(r)
qc3.z(r)
qc3.t(r)

qc3.reset(r[0])

for i in range(n-1):
    qc.cx(q[i], q[i+1])
    # qc.cx(q[i], q[i])  # raise exception

for i in range(n):
    qc.u1(math.pi / (i+1), q[i])
    qc.h(q[i])
    qc.measure(q[i], c[i])
qc.ccx(q[0], q[1], q[2])
print(qc.qasm())
print(qc3.qasm())

q1 = QuantumRegister("q", n)
c1 = ClassicalRegister("c", n)
tqc1 = QuantumCircuit(q1, c1)
# q1 = QuantumRegister("qq", n)  # this raises exception
# q1 = QuantumRegister("q", n-1)  # this raises exception
tqc2 = QuantumCircuit(q1)
tqc2.barrier(q1)
tqc1.s(q1[0]).inverse()
tqc2.s(q1[0])
tqc2.ccx(q1[0], q1[1], q1[2])
tqc1.measure(q1[0], c1[0]).c_if(c1, 3)
tqc1 += tqc2
tqc3 = tqc1 + tqc2
print(tqc1.qasm())
print(tqc3.qasm())
