import sys
import os

# We don't know from where the user is running the example,
# so we need a relative position from this file path.
# TODO: Relative imports for intra-package imports are highly discouraged.
# http://stackoverflow.com/a/7506006
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
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
