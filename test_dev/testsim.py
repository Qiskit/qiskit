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

qp = QuantumProgram()
qp.load_qasm("example", qasm_file="example.qasm")

print('using the unitary simulator')
a = UnitarySimulator(qp.get_qasm("example")).run()
print('\n\n state from unitary = ')

quantum_state = np.zeros(2**(a['number_of_qubits']), dtype=complex)
quantum_state[0] = 1

print(np.dot(a['result']['data']['unitary'], quantum_state))

print('\n\nusing the qasm simulator')
shots = 1024
outcomes = []
seed = 1
b = QasmSimulator(qp.get_qasm("example"), 1).run()
print(b['result']['data']['quantum_state'])
print(b['result']['data']['classical_state'])

b = QasmSimulator(qp.get_qasm("example"), shots).run()
print(b['result']['data']['counts'])
