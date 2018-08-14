import sys
if sys.version_info < (3,5):
    raise Exception('Please use Python version 3.5 or greater.')
import os
from qiskit import QuantumCircuit, QuantumProgram
import numpy as np
np.set_printoptions(threshold=np.nan)
from os import listdir
from os.path import isfile, join

currentFolder = os.path.dirname(os.path.realpath(__file__))
testcaseFolder = currentFolder + "/testcases"

onlyfiles = [f for f in listdir(testcaseFolder) if isfile(join(testcaseFolder, f)) and f.endswith(".qasm")]



for f in onlyfiles:
    Q_program = QuantumProgram()
    currentFolder = os.path.dirname(os.path.realpath(__file__))
    print("analyzing: " + testcaseFolder+"/"+f )
    myqasm = Q_program.load_qasm_file(testcaseFolder+"/"+f, "my_example")
    circuits = ['my_example'] #, 'superposition'
    backend = 'local_sympy_unitary_simulator' # the device to run on
    result = Q_program.execute(circuits, backend=backend, timeout=10)
    print("unitary matrix of the circuit: ")
    print(result.get_data('my_example')['unitary'])
    print("\n")

# array([[sqrt(2)/2, sqrt(2)/2, 0, 0],
#        [0, 0, sqrt(2)/2, -sqrt(2)/2],
#        [0, 0, sqrt(2)/2, sqrt(2)/2],
#        [sqrt(2)/2, -sqrt(2)/2, 0, 0]], dtype=object)

