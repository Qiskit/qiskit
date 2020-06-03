from qiskit.circuit.quantumcircuit import QuantumCircuit


def template_6a_1():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.cx(1, 0)
    qc.cx(0, 1)
    qc.cx(1, 0)
    qc.cx(0, 1)
    qc.cx(1, 0)
    return qc
