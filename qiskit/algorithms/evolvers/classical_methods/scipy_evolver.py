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
"""Parent class for SciPyEvolvers"""
from abc import ABC
from typing import List, Tuple, Union

import scipy.sparse as sp
import numpy as np

from ...list_or_dict import ListOrDict


class SciPyEvolver(ABC):
    """Parent class for SciPyEvolvers"""
    def _create_observable_output(
        self,
        ops_ev_mean: np.ndarray,
        ops_ev_std: np.ndarray,
        aux_ops: ListOrDict,
    ) -> ListOrDict[Union[Tuple[np.ndarray, np.ndarray], Tuple[complex, complex]]]:
        """Creates the right output format for the evaluated auxiliary operators."""
        operator_number = 0 if aux_ops is None else len(aux_ops)
        observable_evolution = [(ops_ev_mean[i], ops_ev_std[i]) for i in range(operator_number)]

        if isinstance(aux_ops, dict):
            observable_evolution = dict(zip(aux_ops.keys(), observable_evolution))

        return observable_evolution

    def _evaluate_aux_ops(
        self,
        aux_ops: List[sp.csr_matrix],
        aux_ops_2: List[sp.csr_matrix],
        state: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluates the aux operators if they are provided and stores their value.

        Returns:
            Tuple of the mean and standard deviation of the aux operators for a given state.
        """
        op_mean = np.array([np.real(state.conjugate().dot(op.dot(state))) for op in aux_ops])
        op_std = np.sqrt(
            np.array([np.real(state.conjugate().dot(op2.dot(state))) for op2 in aux_ops_2])
            - op_mean**2
        )
        return op_mean, op_std
