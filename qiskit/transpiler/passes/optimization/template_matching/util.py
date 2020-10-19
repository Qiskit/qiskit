import numpy as np
from itertools import zip_longest

from qiskit.circuit import Gate, ParameterExpression


def soft_compare(right: Gate, left: Gate) -> bool:
    """
    Soft comparison between gates. Their names, number of qubits, and classical
    bit numbers must match.
    """
    _CUTOFF_PRECISION = 1e-10

    if right.name != left.name or right.num_qubits != left.num_qubits or right.num_clbits != left.num_clbits:
        return False

    for self_param, other_param in zip_longest(right.params, left.params):
        try:
            if self_param == other_param:
                continue
        except ValueError:
            pass

        if not isinstance(self_param, ParameterExpression) and not isinstance(
                other_param, ParameterExpression):
            try:
                if np.shape(self_param) == np.shape(other_param) \
                        and np.allclose(self_param, other_param,
                                        atol=_CUTOFF_PRECISION):
                    continue
            except TypeError:
                pass

            try:
                if np.isclose(float(self_param), float(other_param),
                              atol=_CUTOFF_PRECISION):
                    continue
            except TypeError:
                pass

        else:
            try:
                if np.shape(self_param) == np.shape(other_param):
                    continue
            except TypeError:
                pass

        return False

    return True
