from qiskit.circuit.quantumcircuit import QuantumCircuit


def template_9c_10():
    qc = QuantumCircuit(3)
    qc.ccx(0, 2, 1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.ccx(0, 2, 1)
    qc.cx(0, 2)
    qc.ccx(0, 2, 1)
    qc.cx(1, 2)
    qc.cx(0, 2)
    qc.ccx(0, 2, 1)
    return qc