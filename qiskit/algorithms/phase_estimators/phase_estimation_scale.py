# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Scaling for Hamiltonian and eigenvalues to avoid phase wrapping"""
from __future__ import annotations
import numpy as np

from qiskit.opflow import SummedOp, PauliSumOp
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit.quantum_info.operators.base_operator import BaseOperator


class PhaseEstimationScale:
    """Set and use a bound on eigenvalues of a Hermitian operator in order to ensure phases are in
    the desired range and to convert measured phases into eigenvectors.

    The ``bound`` is set when constructing this class. Then the method ``scale`` is used to find the
    factor by which to scale the operator.

    If ``bound`` is equal exactly to the largest eigenvalue, and the smallest eigenvalue is minus
    the largest, then these two eigenvalues will not be distinguished. For example, if the Hermitian
    operator is the Pauli Z operator with eigenvalues :math:`1` and :math:`-1`, and ``bound`` is
    :math:`1`, then both eigenvalues will be mapped to :math:`1`.
    This can be avoided by making ``bound`` a bit larger.

    Increasing ``bound`` decreases the part of the interval :math:`[0, 1)` that is used to map
    eigenvalues to ``phi``. However, sometimes this results in a better determination of the
    eigenvalues, because 1) although there are fewer discrete phases in the useful range, it may
    shift one of the discrete phases closer to the actual phase. And, 2) If one of the discrete
    phases is close to, or exactly equal to the actual phase, then artifacts (probability) in
    neighboring phases will be reduced.  This is important because the artifacts may be larger than
    the probability in a phase representing another eigenvalue of interest whose corresponding
    eigenstate has a relatively small weight in the input state.

    """

    def __init__(self, bound: float) -> None:
        """
        Args:
            bound: an upper bound on the absolute value of the eigenvalues of a Hermitian operator.
                (The operator is not needed here.)
        """
        self._bound = bound

    @property
    def scale(self) -> float:
        r"""Return the Hamiltonian scaling factor.

        Return the scale factor by which a Hermitian operator must be multiplied
        so that the phase of the corresponding unitary is restricted to :math:`[-\pi, \pi]`.
        This factor is computed from the bound on the absolute values of the eigenvalues
        of the operator. The methods ``scale_phase`` and ``scale_phases`` are used recover
        the eigenvalues corresponding the original (unscaled) Hermitian operator.

        Returns:
            The scale factor.
        """
        return np.pi / self._bound

    def scale_phase(self, phi: float, id_coefficient: float = 0.0) -> float:
        r"""Convert a phase into an eigenvalue.

        The input phase ``phi`` corresponds to the eigenvalue of a unitary obtained by
        exponentiating a scaled Hermitian operator. Recall that the phase
        is obtained from ``phi`` as :math:`2\pi\phi`. Furthermore, the Hermitian operator
        was scaled so that ``phi`` is restricted to :math:`[-1/2, 1/2]`, corresponding to
        phases in :math:`[-\pi, \pi]`. But the values of `phi` read from the phase-readout
        register are in :math:`[0, 1)`. Any value of ``phi`` greater than :math:`1/2` corresponds
        to a raw phase of minus the complement with respect to 1. After this possible
        shift, the phase is scaled by the inverse of the factor by which the
        Hermitian operator was scaled to recover the eigenvalue of the Hermitian
        operator.

        Args:
            phi: Normalized phase in :math:`[0, 1)` to be converted to an eigenvalue.
            id_coefficient: All eigenvalues are shifted by this value.

        Returns:
            An eigenvalue computed from the input phase.
        """
        w = 2 * self._bound
        if phi <= 0.5:
            return phi * w + id_coefficient
        else:
            return (phi - 1) * w + id_coefficient

    def scale_phases(self, phases: list | dict, id_coefficient: float = 0.0) -> dict | list:
        """Convert a list or dict of phases to eigenvalues.

        The values in the list, or keys in the dict, are values of ``phi` and
        are converted as described in the description of ``scale_phase``. In case
        ``phases`` is a dict, the values of the dict are passed unchanged.

        Args:
            phases: a list or dict of values of ``phi``.
            id_coefficient: All eigenvalues are shifted by this value.

        Returns:
            Eigenvalues computed from phases.
        """
        if isinstance(phases, list):
            phases = [self.scale_phase(x, id_coefficient) for x in phases]
        else:
            phases = {self.scale_phase(x, id_coefficient): phases[x] for x in phases.keys()}

        return phases

    @classmethod
    def from_pauli_sum(
        cls, pauli_sum: SummedOp | PauliSumOp | SparsePauliOp | Operator
    ) -> "PhaseEstimationScale" | float:
        """Create a PhaseEstimationScale from a `SummedOp` representing a sum of Pauli Operators.

        It is assumed that the ``pauli_sum`` is the sum of ``PauliOp`` objects. The bound on
        the absolute value of the eigenvalues of the sum is obtained as the sum of the
        absolute values of the coefficients of the terms. This is the best bound available in
        the generic case. A ``PhaseEstimationScale`` object is instantiated using this bound.

        Args:
            pauli_sum: A ``SummedOp`` whose terms are ``PauliOp`` objects.

        Raises:
            ValueError: if ``pauli_sum`` is not a sum of Pauli operators.

        Returns:
            A ``PhaseEstimationScale`` object
        """
        if isinstance(pauli_sum, PauliSumOp):
            bound = abs(pauli_sum.coeff) * sum(abs(coeff) for coeff in pauli_sum.coeffs)
            return PhaseEstimationScale(bound)
        elif isinstance(pauli_sum, SparsePauliOp):
            bound = sum(abs(coeff) for coeff in pauli_sum.coeffs)
            return PhaseEstimationScale(bound)
        elif isinstance(pauli_sum, Operator):
            bound = np.sum(np.abs(np.linalg.eigvalsh(pauli_sum)))
            return PhaseEstimationScale(bound)
        elif isinstance(pauli_sum, BaseOperator):
            raise ValueError(
                f"For the operator of type {type(pauli_sum)} the bound needs to be provided in the "
                f"algorithm."
            )
        else:
            if pauli_sum.primitive_strings() != {"Pauli"}:
                raise ValueError(
                    "`pauli_sum` must be a sum of Pauli operators. Got primitives {}.".format(
                        pauli_sum.primitive_strings()
                    )
                )

            bound = abs(pauli_sum.coeff) * sum(abs(pauli.coeff) for pauli in pauli_sum)
        return PhaseEstimationScale(bound)
