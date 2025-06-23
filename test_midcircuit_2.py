import io

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, Measure
from qiskit.circuit.singleton import SingletonInstruction, stdlib_singleton_key
from qiskit.circuit.exceptions import CircuitError
from qiskit._accelerate.circuit import StandardInstructionType
from qiskit.qpy import dump, load
from qiskit.transpiler.passes import ResetAfterMeasureSimplification


class CustomMeasurement(Measure):
    """A custom specialized measurement."""

    def __init__(self, label=None):
        super().__init__(label=label)
        if label:
            self.name= label
        else:
            self.name="measure_2"

def create():
    circ = QuantumCircuit(2, 2)
    circ.append(CustomMeasurement(), [0], [0])
    circ.append(CustomMeasurement("measure_3"), [0], [1])
    circ.measure_all()
    print(circ.draw())
    print(circ.data)
    print("Is subclass", issubclass(CustomMeasurement, Measure))
    print("Is instance", isinstance(circ.data[0].operation, Measure))
    return circ

def qpy_roundtrip(circ):
    print("QPY ROUNDTRIP")
    with io.BytesIO() as f:
        dump(circ, f)
        f.seek(0)
        out_circ = load(f)[0]
    print(out_circ.draw())
    print(out_circ.data)
    return out_circ

def test_roundtrip(circ, out_circ):
    print("ARE THEY EQUAL?")
    for i, out_item in enumerate(out_circ.data):
        print(out_item.name, "==", circ.data[i].name, "?", out_item == circ.data[i])
        # if not out_item == circ.data[i]:
        print("Before: ",isinstance(circ.data[i].operation, Measure), type(circ.data[i].operation).__mro__)
        print(circ.data[i].operation.definition)
        print("After: ", isinstance(out_item.operation, Measure), type(out_item.operation).__mro__)
        print(out_item.operation.definition)


def qasm3_roundtrip(circ):
    import qiskit.qasm3 as qasm3
    print("QASM3 ROUNDTRIP")
    qasm_str = qasm3.dumps(circ)
    circ_qasm = qasm3.loads(qasm_str)
    return circ_qasm


qc = create()
qpy_qc = qpy_roundtrip(qc)
test_roundtrip(qc, qpy_qc)
qasm3_qc = qasm3_roundtrip(qc)
test_roundtrip(qc, qasm3_qc)
