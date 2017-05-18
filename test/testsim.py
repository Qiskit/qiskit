"""Quick test program for unitary and qasm simulator backend.

Author: Jay Gambetta
"""
import qiskit.unroll as unroll
from qiskit.qasm import Qasm
from qiskit.unroll import SimulatorBackend
from qiskit.simulators._unitarysimulator import UnitarySimulator
from qiskit.simulators._qasmsimulator import QasmSimulator
import random
from collections import Counter
import numpy as np

basis = []  # empty basis, defaults to U, CX
unroller = unroll.Unroller(Qasm(filename="example.qasm").parse(),
                           SimulatorBackend(basis))
unroller.backend.set_trace(False)  # print calls as they happen
unroller.execute()  # Here is where simulation happens


print('using the unirary simulator')
a = UnitarySimulator(unroller.backend.circuit).run()
print('\n\n state from unitary = ')

quantum_state = np.zeros(2**(a['number_of_qubits']), dtype=complex)
quantum_state[0] = 1

print(np.dot(a['result']['data']['unitary'], quantum_state))

print('\n\nusing the qasm simulator')
shots = 1024
outcomes = []
seed = 1
b = QasmSimulator(unroller.backend.circuit, 1).run()
print(b['result']['data']['quantum_state'])
print(b['result']['data']['classical_state'])


print('\n\nrunning many shots')
for i in range(shots):
    # running the quantum_circuit
    b = QasmSimulator(unroller.backend.circuit, random.random()).run()
    # print(b['resuls']['data']['quantum_state'])
    # print(b['resuls']['data']['classical_state'])
    outcomes.append(bin(b['result']['data']['classical_state'])[2:].zfill(b['number_of_cbits']))

print('\n\n outcomes = ')
print(dict(Counter(outcomes)))
