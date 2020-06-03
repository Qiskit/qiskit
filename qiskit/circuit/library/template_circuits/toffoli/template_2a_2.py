from qiskit.circuit.quantumcircuit import QuantumCircuit


def template_2a_2():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.cx(0, 1)
    return qc
