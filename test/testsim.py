"""Quick test program for unitary simulator backend."""
from qiskit.simulators import UnitarySimulator
basis = []  # empty basis, defaults to U, CX
unroller = unroll.Unroller(Qasm(filename="example.qasm").parse(),
                           unroll.UnitarySimulator(basis))
unroller.backend.set_trace(True)  # print calls as they happen
unroller.execute()  # Here is where simulation happens
