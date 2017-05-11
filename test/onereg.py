"""Quick test program for "one register" backend."""
import qiskit.unroll as unroll
from qiskit.qasm import Qasm
fname = "../examples/Python/qasm_examples/adder.qasm"
basis = []  # empty basis, defaults to U, CX
unroller = unroll.Unroller(Qasm(filename=fname).parse(),
                           unroll.OneRegisterBackend(basis))
unroller.execute()
print(unroller.backend.circuit.qasm(qeflag=True))
