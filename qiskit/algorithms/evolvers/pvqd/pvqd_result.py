# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Result object for p-VQD."""

from typing import Union, Optional, List, Tuple
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.opflow import StateFn, OperatorBase

from ..evolution_result import EvolutionResult


class PVQDResult(EvolutionResult):
    """The result object for the p-VQD algorithm."""

    def __init__(
        self,
        evolved_state: Union[StateFn, QuantumCircuit, OperatorBase],
        aux_ops_evaluated: Optional[List[Tuple[complex, complex]]] = None,
        times: Optional[List[float]] = None,
        parameters: Optional[List[np.ndarray]] = None,
        fidelities: Optional[List[float]] = None,
        estimated_error: Optional[float] = None,
        observables: Optional[List[List[float]]] = None,
    ):
        """
        Args:
            evolved_state: An evolved quantum state.
            aux_ops_evaluated: Optional list of observables for which expected values on an evolved
                state are calculated. These values are in fact tuples formatted as (mean, standard
                deviation).
            times: The times evaluated during the time integration.
            parameters: The parameter values at each evaluation time.
            fidelities: The fidelity of the Trotter step and variational update at each iteration.
            estimated_error: The overall estimated error evaluated as one minus the
                product of all fidelities.
            observables: The value of the observables evaluated at each iteration.
        """
        super().__init__(evolved_state, aux_ops_evaluated)
        self.times = times
        self.parameters = parameters
        self.fidelities = fidelities
        self.estimated_error = estimated_error
        self.observables = observables
