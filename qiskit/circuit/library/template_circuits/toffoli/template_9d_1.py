from qiskit.circuit.quantumcircuit import QuantumCircuit


def template_9d_1():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.x(1)
    qc.cx(1, 0)
    qc.x(1)
    qc.cx(0, 1)
    qc.cx(1, 0)
    qc.x(1)
    qc.cx(0, 1)
    qc.cx(1, 0)
    return qc
