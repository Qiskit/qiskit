import sys
if sys.version_info < (3,5):
    raise Exception('Please use Python version 3.5 or greater.')

# useful additional packages
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as la

# importing the QISKit
from qiskit import QuantumCircuit, QuantumProgram

from qiskit.tools.visualization import plot_histogram, plot_state

Q_program = QuantumProgram()
n = 3  # number of qubits
q = Q_program.create_quantum_register('q', n)
c = Q_program.create_classical_register('c', n)

# quantum circuit to make a GHZ state
ghz = Q_program.create_circuit('ghz', [q], [c])
ghz.h(q[0])
ghz.cx(q[0], q[1])
ghz.cx(q[0], q[2])
ghz.s(q[0])
ghz.measure(q[0], c[0])
ghz.measure(q[1], c[1])
ghz.measure(q[2], c[2])

# quantum circuit to make a superpostion state
superposition = Q_program.create_circuit('superposition', [q], [c])
superposition.h(q)
superposition.s(q[0])
superposition.measure(q[0], c[0])
superposition.measure(q[1], c[1])
superposition.measure(q[2], c[2])

circuits = ['ghz', 'superposition']

# execute the quantum circuit
backend = 'local_qasm_simulator' # the device to run on
result = Q_program.execute(circuits, backend=backend, shots=1000)

#{'000': 494, '111': 506}
#{'010': 135, '111': 102, '101': 134, '000': 120, '100': 135, '001': 129, '110': 120, '011': 125}

print(result.get_counts('ghz'))

print(result.get_counts('superposition'))

#plot_histogram(result.get_counts('ghz'))
#plot_histogram(result.get_counts('superposition'),15)
