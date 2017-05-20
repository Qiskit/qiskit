"""Quick test program for unitary and qasm simulator backend.

Author: Jay Gambetta
"""
import sys
sys.path.append("..")
from qiskit import QuantumProgram
import qiskit.unroll as unroll
from qiskit.qasm import Qasm
from qiskit.unroll import SimulatorBackend
from qiskit.simulators._unitarysimulator import UnitarySimulator
from qiskit.simulators._qasmsimulator import QasmSimulator
import random
from collections import Counter
import numpy as np

# to make command line imputs
simulator_to_run = "qasm_simulator"

#simulator_to_run = "unitary_simulator"
seed = 1

if len(sys.argv) != 2:
    print("testsim <filename.qasm>")
    sys.exit(1)

filename = sys.argv[1]

qp = QuantumProgram()
qp.load_qasm("example", qasm_file=filename)


if simulator_to_run == "unitary_simulator":
    print('using the unitary simulator')
    a = UnitarySimulator(qp.get_qasm("example")).run()
    dim = len(a['data']['unitary'])
    print('\n\n state from unitary = ')
    quantum_state = np.zeros(dim, dtype=complex)
    quantum_state[0] = 1
    print(np.dot(a['data']['unitary'], quantum_state))
    print(b['status'])

if simulator_to_run == "qasm_simulator_single_shot":
    print('using the qasm simulator in single shot mode \n ')
    b = QasmSimulator(qp.get_qasm("example"), seed).run()
    print(b['data']['quantum_state'])
    print(b['data']['classical_state'])
    print(b['status'])

if simulator_to_run == "qasm_simulator":
    print('using the qasm simulator \n ')
    shots = 1024
    b = QasmSimulator(qp.get_qasm("example"), shots).run()
    print(b['data']['counts'])
    print(b['status'])
