from qiskit.circuit.quantumcircuit import QuantumCircuit


def template_5a_1():
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    qc.cx(0, 1)
    qc.ccx(0, 1, 2)
    qc.cx(0, 1)
    qc.cx(0, 2)
    return qc
