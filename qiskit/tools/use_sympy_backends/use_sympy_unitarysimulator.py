import sys
import os
from qiskit import QuantumProgram
import numpy as np
np.set_printoptions(threshold=np.nan)

Q_program = QuantumProgram()
currentFolder = os.path.dirname(os.path.realpath(__file__))
qasm_file = currentFolder + "/../../../examples/qasm/naive.qasm"
myqasm = Q_program.load_qasm_file(qasm_file, "my_example")
print("analyzing: " + qasm_file)
circuits = ['my_example']
backend = 'local_sympy_unitary_simulator'
result = Q_program.execute(circuits, backend=backend, timeout=10)
print("unitary matrix of the circuit: ")
print(result.get_data('my_example')['unitary'])
