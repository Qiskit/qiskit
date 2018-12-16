from qiskit import *
from qiskit.tools.visualization import *
from qiskit.converters import circuit_to_dag

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

dag = circuit_to_dag(circuit)
circuit.draw(interactive=True, output='latex')

dag_drawer(dag)

pm = PassManager()

pm.append([CommutationAnalysis(), CommutationTransformation()])

dag = transpile_dag(dag, pass_manager=pm)
dag_drawer(dag)
