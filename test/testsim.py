"""Quick test program for unitary simulator backend."""
import qiskit.unroll as unroll
from qiskit.qasm import Qasm
from qiskit.unroll import SimulatorBackend
from qiskit.simulators._unitarysimulator import UnitarySimulator
from qiskit.simulators._qasmsimulator import QasmSimulator
import random
from collections import Counter

basis = []  # empty basis, defaults to U, CX
unroller = unroll.Unroller(Qasm(filename="example.qasm").parse(),
                           SimulatorBackend(basis))
unroller.backend.set_trace(False)  # print calls as they happen
unroller.execute()  # Here is where simulation happens


print('using the unirary simulator')
a = UnitarySimulator(unroller.backend.circuit).run()
print('\n\n Unitary = ')
print(a['result']['unitary'])

print('\n\nusing the qasm simulator')
shots = 1024
outcomes = []
for i in range(shots):
    # running the quantum_circuit
    b = QasmSimulator(unroller.backend.circuit, random.random()).run()
    #print(b['result']['quantum_state'])
    #print(b['result']['classical_state'])
    outcomes.append(bin(b['result']['classical_state'])[2:].zfill(b['number_of_cbits']))

print('\n\n outcomes = ')
print(dict(Counter(outcomes)))
