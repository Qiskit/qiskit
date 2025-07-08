import io

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, Measure
from qiskit.circuit.singleton import SingletonInstruction, stdlib_singleton_key
from qiskit.circuit.exceptions import CircuitError
from qiskit._accelerate.circuit import StandardInstructionType
from qiskit.qpy import dump, load
from qiskit.transpiler.passes import ResetAfterMeasureSimplification


# IDEA 1: Subclass Measure
# Pitfall: can't change name
class MidCircuitMeasureSubclass(Measure):
    pass

# IDEA 1.2: Subclass Measure
class CustomMeasurement(Measure):
    """A custom specialized measurement."""

    def __init__(self, label=None):
        super().__init__(label=label)
        if label:
            self.name= label
        else:
            self.name="measure_2"


# IDEA 2: Just build an instruction
# Pitfall: the transpiler doesn't detect it as a measurement
class MidCircuitMeasureInstruction(Instruction):
    def __init__(self, label=None):
        super().__init__("mid_circuit_measure", 1, 1, [], label=label)


# IDEA 3: copy Measure() definition. Works!
# Pitfall:
# - setting the standard instruction type like this looks a bit sketchy
# - if we do this from outside qiskit (qiskit-ibm-runtime), we'd have to import 
#    StandardInstructionType from qiskit._accelerate.circuit
class NamedMeasure(SingletonInstruction):
    """Quantum measurement in the computational basis."""

    # Just force the standard instruction type?
    _standard_instruction_type = StandardInstructionType.Measure

    def __init__(self, name="measure_2", label=None):
        """
        Args:
            label: optional string label for this instruction.
        """
        super().__init__(name, 1, 1, [], label=label)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Subclasses of Measure are not "standard", so we set this to None to
        # prevent the Rust code from treating them as such.
        cls._standard_instruction_type = None

    _singleton_lookup_key = stdlib_singleton_key()

    def broadcast_arguments(self, qargs, cargs):
        qarg = qargs[0]
        carg = cargs[0]

        if len(carg) == len(qarg):
            for qarg, carg in zip(qarg, carg):
                yield [qarg], [carg]
        elif len(qarg) == 1 and carg:
            for each_carg in carg:
                yield qarg, [each_carg]
        else:
            raise CircuitError("register size error")

loaded = {}
for cls in [CustomMeasurement]:
    circ = QuantumCircuit(2, 2)
    circ.append(cls(), [0], [0])
    circ.append(cls("measure_3"), [0], [1])
    circ.measure_all()
    print(circ.draw())

    with io.BytesIO() as f:
        dump(circ, f)
        f.seek(0)
        loaded[str(cls)] = load(f)


for cls, loaded_circ in loaded.items():
    print(cls)
    print(list(loaded_circ[0].data))

def test_bv_circuit():
    """Test Bernstein Vazirani circuit with midcircuit measurement."""
    bitstring = "11111"
    qc = QuantumCircuit(2, len(bitstring))
    qc.x(1)
    qc.h(1)
    for idx, bit in enumerate(bitstring[::-1]):
        qc.h(0)
        if int(bit):
            qc.cx(0, 1)
        qc.h(0)
        qc.append(CustomMeasurement(label="measure_3"), [0], [idx])
        # qc.measure(0, idx)
        if idx != len(bitstring) - 1:
            qc.reset(0)
            # reset control
            qc.reset(1)
            qc.x(1)
            qc.h(1)
    print(qc.draw())
    new_qc = ResetAfterMeasureSimplification()(qc)
    print(new_qc.draw())

    for op in new_qc.data:
        if op.operation.name == "reset":
            print(op.qubits[0] == new_qc.qubits[1])
            # self.assertEqual(op.qubits[0], new_qc.qubits[1])

# test_bv_circuit()