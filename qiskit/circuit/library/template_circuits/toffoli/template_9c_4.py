from qiskit.circuit.quantumcircuit import QuantumCircuit


def template_9c_4():
    qc = QuantumCircuit(3)
    qc.ccx(0, 2, 1)
    qc.ccx(0, 1, 2)
    qc.x(1)
    qc.ccx(0, 2, 1)
    qc.x(2)
    qc.cx(2, 1)
    qc.ccx(0, 1, 2)
    qc.x(2)
    qc.cx(2, 1)
    return qc
