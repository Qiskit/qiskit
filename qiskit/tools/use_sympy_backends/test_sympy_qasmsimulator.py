import sys
if sys.version_info < (3,5):
    raise Exception('Please use Python version 3.5 or greater.')
import os
from qiskit import QuantumCircuit, QuantumProgram

from os import listdir
from os.path import isfile, join
currentFolder = os.path.dirname(os.path.realpath(__file__))
testcaseFolder = currentFolder + "/testcases"

onlyfiles = [f for f in listdir(testcaseFolder) if isfile(join(testcaseFolder, f)) and f.endswith(".qasm")]

for f in onlyfiles:
    Q_program = QuantumProgram()
    print("analyzing: " + testcaseFolder+"/"+f )
    myqasm = Q_program.load_qasm_file(testcaseFolder+"/"+f, "my_example")
    circuits = ['my_example'] #, 'superposition'
    backend = 'local_sympy_qasm_simulator' # the device to run on
    result = Q_program.execute(circuits, backend=backend, shots=10, timeout=300)
    print("count:")
    print(result.get_counts('my_example'))
    print("quantum_state prior to measurement: ")
    print(result.get_data('my_example')['quantum_state'])
    print("\n")



#from qiskit.tools.visualization import plot_histogram, plot_state
#plot_histogram(result.get_counts('my_example'))
