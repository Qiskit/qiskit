from qiskit import *
from qiskit.tools.visualization import *

q = QuantumRegister(3)
c = ClassicalRegister(2)
circ = QuantumCircuit(q, c)
circ.x(q[0])
circ.h(q[1])
circ.cx(q[1], q[2])
circuit_drawer(circ)
