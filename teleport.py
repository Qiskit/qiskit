"""
Quantum teleportation example based on OPENQASM example.

Author: Andrew Cross
"""
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.extensions.standard import barrier, h, cx, u3, x, z

# define Program
# add circuit to Program
# add registers to circuit

q = QuantumRegister("q", 3)
c0 = ClassicalRegister("c0", 1)
c1 = ClassicalRegister("c1", 1)
c2 = ClassicalRegister("c2", 1)
qc = QuantumCircuit(q, c0, c1, c2)
qc.u3(0.3, 0.2, 0.1, q[0])
qc.h(q[1])
qc.cx(q[1], q[2])
qc.barrier(q)

qc.cx(q[0], q[1])
qc.h(q[0])
qc.measure(q[0], c0[0])
qc.measure(q[1], c1[0])

qc.z(q[2]).c_if(c0, 1)
qc.x(q[2]).c_if(c1, 1)
qc.measure(q[2], c2[0])

print(qc.qasm())
