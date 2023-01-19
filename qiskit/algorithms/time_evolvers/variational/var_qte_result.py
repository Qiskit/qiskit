# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Result object for varQTE."""
from __future__ import annotations

import numpy as np

from qiskit.circuit import QuantumCircuit

from ..time_evolution_result import TimeEvolutionResult

from ...list_or_dict import ListOrDict


class VarQTEResult(TimeEvolutionResult):
    """The result object for the variational quantum time evolution algorithms.

    Attributes:
        parameter_values (np.array | None): Optional list of parameter values obtained after
            each evolution step.
    """

    def __init__(
        self,
        evolved_state: QuantumCircuit,
        aux_ops_evaluated: ListOrDict[tuple[complex, complex]] | None = None,
        observables: ListOrDict[tuple[np.ndarray, np.ndarray]] | None = None,
        times: np.ndarray | None = None,
        parameter_values: np.ndarray | None = None,
    ):
        """
        Args:
            evolved_state: An evolved quantum state.
            aux_ops_evaluated: Optional list of observables for which expected values on an evolved
                state are calculated. These values are in fact tuples formatted as (mean, standard
                deviation).
            observables: Optional list of observables for which expected on an evolved state are
                calculated at each timestep.
                These values are in fact lists of tuples formatted as (mean, standard deviation).
            times: Optional list of times at which each observable has been evaluated.
            parameter_values: Optional list of parameter values obtained after each evolution step.

        """

        super().__init__(evolved_state, aux_ops_evaluated, observables, times)
        self.parameter_values = parameter_values
