from qiskit.circuit.quantumcircuit import QuantumCircuit


def template_5a_4():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.x(0)
    qc.cx(0, 1)
    qc.x(0)
    qc.x(1)
    return qc