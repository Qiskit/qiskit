from qiskit.circuit.quantumcircuit import QuantumCircuit

def template_4b_2():
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    qc.cx(1, 2)
    qc.ccx(0, 1, 2)
    qc.cx(1, 2)
    return qc