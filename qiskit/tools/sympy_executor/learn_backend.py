import sys
if sys.version_info < (3,5):
    raise Exception('Please use Python version 3.5 or greater.')

from qiskit import QuantumCircuit, QuantumProgram



Q_program = QuantumProgram()
ghz = Q_program.load_qasm_file("/Users/liup/quantum/qiskit-pull-requests/qiskit/tools/sympy_executor/testcases/naive.qasm", "ghz")



circuits = ['ghz'] #, 'superposition'

# execute the quantum circuit
backend = 'local_sympy_qasm_simulator' # the device to run on
result = Q_program.execute(circuits, backend=backend, shots=10)

print(result.get_counts('ghz'))

# print(result.get_counts('superposition'))

#plot_histogram(result.get_counts('ghz'))
#plot_histogram(result.get_counts('superposition'),15)
