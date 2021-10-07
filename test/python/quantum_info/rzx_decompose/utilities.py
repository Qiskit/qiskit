"""
Helper functions for the RZXDecomposer test suite.
"""

import numpy as np

from qiskit.circuit.library import RXXGate, RYYGate, RZZGate

def canonical_matrix(a=0.0, b=0.0, c=0.0):
    return (
        RXXGate(2 * a).to_matrix() @
        RYYGate(2 * b).to_matrix() @
        RZZGate(-2 * c).to_matrix()
    )
