import io

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, Measure
import qiskit.qasm3 as qasm3
from qiskit.qasm3 import DefcalInstruction
from qiskit.circuit.classical import types

from qiskit.circuit import Instruction

class MidCircuitMeasure(Instruction):
    def __init__(self, name="measure_2", label=None):
        super().__init__(name, 1, 1, [], label=label)

qc = QuantumCircuit(2, 2)
qc.append(MidCircuitMeasure(), [0], [0])
qc.append(MidCircuitMeasure("measure_3"), [0], [1])
qc.measure_all()
print(qc.draw())

defcals = {
    "measure_2": DefcalInstruction("measure_2", 0, 1, types.Bool()),
    "measure_3": DefcalInstruction("measure_3", 0, 1, types.Bool()),
}
qasm_str = qasm3.dumps(qc, implicit_defcals=defcals)

print("OUTPUT STRING:\n")
print(qasm_str)