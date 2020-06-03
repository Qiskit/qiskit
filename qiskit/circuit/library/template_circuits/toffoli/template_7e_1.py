from qiskit.circuit.quantumcircuit import QuantumCircuit


def template_7e_1():
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.ccx(0, 2, 1)
    qc.x(2)
    qc.cx(0, 2)
    qc.ccx(0, 2, 1)
    qc.x(0)
    qc.cx(0, 2)
    return qc
