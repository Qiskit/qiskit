from qiskit.circuit.quantumcircuit import QuantumCircuit


def template_5a_3():
    qc = QuantumCircuit(3)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.cx(0, 2)
    return qc
