from qiskit.circuit.quantumcircuit import QuantumCircuit


def template_9d_4():
    qc = QuantumCircuit(3)
    qc.cx(1, 2)
    qc.cx(0, 2)
    qc.cx(2, 1)
    qc.cx(0, 2)
    qc.cx(1, 2)
    qc.cx(2, 1)
    qc.cx(0, 2)
    qc.cx(1, 2)
    qc.cx(2, 1)
    return qc