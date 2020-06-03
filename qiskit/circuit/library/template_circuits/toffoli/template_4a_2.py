from qiskit.circuit.quantumcircuit import QuantumCircuit


def template_4a_2():
    qc = QuantumCircuit(4)
    qc.ccx(0, 1, 3)
    qc.cx(1, 2)
    qc.ccx(0, 1, 3)
    qc.cx(1, 2)
    return qc
