"""
Helper functions for the RZXDecomposer test suite.
"""

from qiskit.circuit.library import RXXGate, RYYGate, RZZGate


def canonical_matrix(a=0.0, b=0.0, c=0.0):
    """
    Produces the matrix form of a "canonical operator"

        exp(-i (a XX + b YY + c ZZ)) .
    """
    return RXXGate(2 * a).to_matrix() @ RYYGate(2 * b).to_matrix() @ RZZGate(-2 * c).to_matrix()
