from qiskit.circuit.quantumcircuit import QuantumCircuit


def template_9a_1():
    qc = QuantumCircuit(3)
    qc.cx(1, 0)
    qc.ccx(0, 2, 1)
    qc.ccx(1, 2, 0)
    qc.x(2)
    qc.cx(0,1)
    qc.ccx(0, 2, 1)
    qc.cx(1, 0)
    qc.x(2)
    qc.ccx(0, 2, 1)
    return qc
