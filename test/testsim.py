"""Quick test program for unitary simulator backend."""
import qiskit.unroll as unroll
from qiskit.qasm import Qasm
from qiskit.unroll import SimulatorBackend
from qiskit.simulators._unitarysimulator import unitary_simulator
basis = []  # empty basis, defaults to U, CX
unroller = unroll.Unroller(Qasm(filename="example.qasm").parse(),
                           SimulatorBackend(basis))
unroller.backend.set_trace(False)  # print calls as they happen
unroller.execute()  # Here is where simulation happens
unitary_gates = unroller.backend.unitary_gates
# print(unitary_gates)
print(unitary_simulator(unitary_gates))
