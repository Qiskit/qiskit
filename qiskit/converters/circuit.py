from qiskit import QuantumCircuit
# from qiskit.dagcircuit import DAGCircuit

def is_circuit(obj):
    """Return `True` if `obj` implements methods that a circuit must."""
    return isinstance(obj, (QuantumCircuit,))

# def is_circuit(obj):
#     """Return `True` if `obj` implements methods that a circuit must."""
#     return isinstance(obj, (QuantumCircuit, DAGCircuit))
