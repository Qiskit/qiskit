from qiskit.circuit.quantumcircuit import QuantumCircuit


def template_4a_5():
    qc = QuantumCircuit(4)
    qc.cx(1, 3)
    qc.ccx(0, 1, 2)
    qc.cx(1, 3)
    qc.ccx(0, 1, 2)
    return qc
