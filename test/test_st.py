"""Test S and T gates, their inverses, and functionality of CompositeGate."""
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


q = QuantumRegister("q", 1)
c = ClassicalRegister("c", 1)
qc = QuantumCircuit(q, c)
qc.s(q[0])
qc.sdg(q[0])
qc.t(q[0])
qc.tdg(q[0])
qc.tdg(q[0]).inverse()
qc.t(q[0]).inverse()
qc.sdg(q[0]).inverse()
qc.s(q[0]).inverse()
qc.s(q[0]).c_if(c, 1).inverse()
qc.s(q[0]).inverse().c_if(c, 1)
qc.t(q[0]).inverse().c_if(c, 1).inverse()
qc.t(q[0]).inverse().inverse().c_if(c, 0)


print(qc.qasm())
