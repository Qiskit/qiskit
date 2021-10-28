"""
qiskit/quantum_info/synthesis/rzx_decompose/utilities.py

Depository for generic utility snippets.
"""

import warnings

import numpy as np


EPSILON = 1e-6  # Fraction(1, 1_000_000)


# TODO: THIS IS A STOPGAP!!!
def safe_arccos(numerator, denominator):
    """
    Computes arccos(n/d) with different (better?) numerical stability.
    """
    threshold = 0.005

    if abs(numerator) > abs(denominator) and abs(numerator - denominator) < threshold:
        return 0.0
    elif abs(numerator) > abs(denominator) and abs(numerator + denominator) < threshold:
        return np.pi
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            return np.arccos(numerator / denominator)
