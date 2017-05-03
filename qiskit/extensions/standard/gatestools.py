
from qiskit import QuantumRegister
from qiskit import InstructionSet

def attach_gate(element, quantum_register ,gate, gate_class):
    if isinstance(quantum_register, QuantumRegister):
        gs = InstructionSet()
        for register in range(quantum_register.size):
            gs.add(gate)
        return gs
    else:
        element._check_qubit(quantum_register)
        return element._attach(gate_class)