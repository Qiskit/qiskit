from qiskit.circuit.quantumcircuit import QuantumCircuit


def template_7a_1():
    qc = QuantumCircuit(3)
    qc.cx(1, 0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.ccx(0, 1, 2)
    qc.cx(0, 1)
    qc.cx(1, 0)
    qc.ccx(0, 1, 2)
    return qc