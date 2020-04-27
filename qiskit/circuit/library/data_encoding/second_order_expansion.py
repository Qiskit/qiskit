# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Second-order Pauli-Z expansion circuit."""

from typing import Callable, List, Union, Optional
import numpy as np
from .pauli_expansion import PauliExpansion


class SecondOrderExpansion(PauliExpansion):
    """Second-order Pauli-Z expansion circuit."""

    def __init__(self,
                 feature_dimension: int,
                 reps: int = 2,
                 entanglement: Union[str, List[List[int]], Callable[[int], List[int]]] = 'full',
                 data_map_func: Optional[Callable[[np.ndarray], float]] = None,
                 insert_barriers: bool = False,
                 ) -> None:
        """Create a new second-order Pauli-Z expansion.

        Args:
            feature_dimension: Number of features.
            reps: The number of repeated circuits, has a min. value of 1.
            entanglement: Specifies the entanglement structure. Refer to
                :class:`~qiskit.circuit.library.NLocal` for detail.
            data_map_func: A mapping function for data x.
            insert_barriers: If True, barriers are inserted in between the evolution instructions
                and hadamard layers.

        """
        super().__init__(feature_dimension=feature_dimension,
                         reps=reps,
                         entanglement=entanglement,
                         paulis=['Z', 'ZZ'],
                         data_map_func=data_map_func)
