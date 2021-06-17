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

"""Result from running HamiltonianPhaseEstimation"""


from typing import Dict, Union, cast
from qiskit.utils.deprecation import deprecate_function
from qiskit.algorithms.algorithm_result import AlgorithmResult
from .phase_estimation_result import PhaseEstimationResult
from .phase_estimation_scale import PhaseEstimationScale


class HamiltonianPhaseEstimationResult(AlgorithmResult):
    """Store and manipulate results from running `HamiltonianPhaseEstimation`.

    This API of this class is nearly the same as `PhaseEstimatorResult`, differing only in
    the presence of an additional keyword argument in the methods. If `scaled`
    is `False`, then the phases are not translated and scaled to recover the
    eigenvalues of the Hamiltonian. Instead `phi` in :math:`[0, 1)` is returned,
    as is the case when then unitary is not derived from a Hamiltonian.

    This class is meant to be instantiated via `HamiltonianPhaseEstimation.estimate`.
    """

    def __init__(
        self,
        phase_estimation_result: PhaseEstimationResult,
        phase_estimation_scale: PhaseEstimationScale,
        id_coefficient: float,
    ) -> None:
        """
        Args:
            phase_estimation_result: The result object returned by PhaseEstimation.estimate.
            phase_estimation_scale: object used to scale phases to obtain eigenvalues.
            id_coefficient: The coefficient of the identity term in the Hamiltonian.
                            Eigenvalues are computed without this term so that the
                            coefficient must added to give correct eigenvalues.
                            This is done automatically when retrieving eigenvalues.
        """
        super().__init__()
        self._phase_estimation_scale = phase_estimation_scale
        self._id_coefficient = id_coefficient
        self._phase_estimation_result = phase_estimation_result

    # pylint: disable=arguments-differ
    def filter_phases(
        self, cutoff: float = 0.0, scaled: bool = True, as_float: bool = True
    ) -> Dict[Union[str, float], float]:
        """Filter phases as does `PhaseEstimatorResult.filter_phases`, with
        the addition that `phi` is shifted and translated to return eigenvalues
        of the Hamiltonian.

        Args:
            cutoff: Minimum weight of number of counts required to keep a bit string.
                The default value is `0.0`.
            scaled: If False, return `phi` in :math:`[0, 1)` rather than the eigenvalues of
                the Hamiltonian.
            as_float: If `True`, returned keys are floats in :math:`[0.0, 1.0)`. If `False`
                returned keys are bit strings.

        Raises:
            ValueError: if `as_float` is `False` and `scaled` is `True`.

        Returns:
            A dict of filtered phases.
        """
        if scaled and not as_float:
            raise ValueError("`as_float` must be `True` if `scaled` is `True`.")

        phases = self._phase_estimation_result.filter_phases(cutoff, as_float=as_float)
        if scaled:
            return cast(
                Dict, self._phase_estimation_scale.scale_phases(phases, self._id_coefficient)
            )
        else:
            return cast(Dict, phases)

    @property
    @deprecate_function(
        """The 'HamiltonianPhaseEstimationResult.most_likely_phase' attribute
                        is deprecated as of 0.18.0 and will be removed no earlier than 3 months
                        after the release date. It has been renamed as the 'phase' attribute."""
    )
    def most_likely_phase(self) -> float:
        """DEPRECATED - The most likely phase of the unitary corresponding to the Hamiltonian.

        Returns:
            The most likely phase.
        """
        return self.phase

    @property
    def phase(self) -> float:
        """The most likely phase of the unitary corresponding to the Hamiltonian.

        Returns:
            The most likely phase.
        """
        return self._phase_estimation_result.phase

    @property
    def most_likely_eigenvalue(self) -> float:
        """The most likely eigenvalue of the Hamiltonian.

        This method calls `most_likely_phase` and scales the result to
        obtain an eigenvalue.

        Returns:
            The most likely eigenvalue of the Hamiltonian.
        """
        phase = self.phase
        return self._phase_estimation_scale.scale_phase(phase, self._id_coefficient)
