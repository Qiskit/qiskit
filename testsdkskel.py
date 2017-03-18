from qiskit_sdk import *
from qiskit_sdk.extensions.standard import h

q = QuantumRegister("q", 5)
c = ClassicalRegister("c", 5)
q.h(0)  # this is added to q but add to p not implemented ...
p = Program(q, c)
q.h(0)  # this gets added to q and p
print(p.qasm())
