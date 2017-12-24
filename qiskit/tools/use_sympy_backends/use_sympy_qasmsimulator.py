import sys
if sys.version_info < (3,5):
    raise Exception('Please use Python version 3.5 or greater.')
import os
from qiskit import QuantumCircuit, QuantumProgram



Q_program = QuantumProgram()
currentFolder = os.path.dirname(os.path.realpath(__file__))
qasm_file = currentFolder + "/testcases/naive.qasm"
myqasm = Q_program.load_qasm_file(qasm_file, "my_example")
print("analyzing: " + qasm_file)
circuits = ['my_example'] #, 'superposition'
backend = 'local_sympy_qasm_simulator' # the device to run on
result = Q_program.execute(circuits, backend=backend, shots=10, timeout=300)
print("count:")
print(result.get_counts('my_example')) #{'11': 54, '00': 46}
print("quantum_state prior to measurement: ")
print(result.get_data('my_example')['quantum_state']) # [sqrt(2)/2 0 0 sqrt(2)/2]
print("\n")




#from qiskit.tools.visualization import plot_histogram, plot_state
#plot_histogram(result.get_counts('my_example'))
