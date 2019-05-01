from qiskit import *

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CommutationAnalysis, CommutativeCancellation

qr = QuantumRegister(5, 'qr')
circuit = QuantumCircuit(qr)
# Quantum Instantaneous Polynomial Time example
circuit.cx(qr[0], qr[1])
circuit.cx(qr[2], qr[1])
circuit.cx(qr[4], qr[3])
circuit.cx(qr[2], qr[3]) 
circuit.z(qr[0])
circuit.z(qr[4])
circuit.cx(qr[0], qr[1])
circuit.cx(qr[2], qr[1])
circuit.cx(qr[4], qr[3])
circuit.cx(qr[2], qr[3]) 
circuit.cx(qr[3], qr[2]) 

print(circuit.draw())

pm = PassManager()
pm.append([CommutationAnalysis(), CommutativeCancellation()])
new_circuit=pm.run(circuit)
print(new_circuit.draw())
