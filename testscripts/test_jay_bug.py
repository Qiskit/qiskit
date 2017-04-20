from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.extensions.standard import h, cx

q2 = QuantumRegister("q", 2)
c2 = ClassicalRegister("c", 2)
bell = QuantumCircuit(q2, c2)
bell.h(q2[0])
bell.cx(q2[0], q2[1])

measureIZ = QuantumCircuit(q2, c2)
measureIZ.measure(q2[0], c2[0])

c = bell + measureIZ

print(c.qasm())
