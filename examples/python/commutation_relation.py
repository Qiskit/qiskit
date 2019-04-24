from qiskit import *

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CommutationAnalysis, CommutativeCancellation
from qiskit.transpiler import transpile

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

# TODO make it not needed to have a backend 
backend_device = BasicAer.get_backend('qasm_simulator')
circuit = transpile(circuit, backend_device, pass_manager=pm)
print(circuit.draw())
