from qiskit.circuit.quantumcircuit import QuantumCircuit


def template_2a_3():
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    qc.ccx(0, 1, 2)
    return qc
