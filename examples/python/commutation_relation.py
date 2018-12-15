from qiskit import *
from qiskit.dagcircuit import DAGCircuit
from qiskit.tools.visualization import *

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CommutationAnalysis, CommutationTransformation
from qiskit.transpiler import transpile_dag

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

qdag = DAGCircuit.fromQuantumCircuit(circuit)
plot_circuit(circuit)

dag_drawer(qdag)

pm = PassManager()

pm.add_passes(CommutationAnalysis())
pm.add_passes(CommutationTransformation())

qdag = transpile_dag(qdag, pass_manager=pm)
dag_drawer(qdag)
