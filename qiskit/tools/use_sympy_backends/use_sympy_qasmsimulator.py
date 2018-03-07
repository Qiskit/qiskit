import sys
import os
from qiskit import QuantumProgram

Q_program = QuantumProgram()
currentFolder = os.path.dirname(os.path.realpath(__file__))
qasm_file = currentFolder + "/../../../examples/qasm/naive.qasm"
myqasm = Q_program.load_qasm_file(qasm_file, "my_example")
print("analyzing: " + qasm_file)
circuits = ['my_example']
backend = 'local_sympy_qasm_simulator'
result = Q_program.execute(circuits, backend=backend, shots=10, timeout=300)
print("final quantum amplitude vector: ")
print(result.get_data('my_example')['quantum_state'])
print("\n")
