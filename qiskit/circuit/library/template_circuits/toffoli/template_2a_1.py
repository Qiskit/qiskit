from qiskit.circuit.quantumcircuit import QuantumCircuit


def template_2a_1():
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.x(0)
    return qc
