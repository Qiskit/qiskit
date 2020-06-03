from qiskit.circuit.quantumcircuit import QuantumCircuit


def template_4a_4():
    qc = QuantumCircuit(3)
    qc.cx(0, 2)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(0, 1)
    return qc
