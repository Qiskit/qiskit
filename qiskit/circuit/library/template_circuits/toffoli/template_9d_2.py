from qiskit.circuit.quantumcircuit import QuantumCircuit


def template_9d_2():
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    qc.cx(0, 2)
    qc.ccx(0, 2, 1)
    qc.x(2)
    qc.cx(1, 2)
    qc.ccx(0, 2, 1)
    qc.x(2)
    qc.cx(1, 2)
    qc.ccx(0, 2, 1)
    return qc
