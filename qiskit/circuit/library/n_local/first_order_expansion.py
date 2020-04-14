# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""First Order Expansion feature map."""

from typing import Callable
import numpy as np
from .pauli_z_expansion import PauliZExpansion
from .data_mapping import self_product


class FirstOrderExpansion(PauliZExpansion):
    """First Order Expansion feature map.

    This is a sub-class of :class:`PauliZExpansion` where *z_order* is fixed at 1.
    As a result the first order expansion will be a feature map without entangling gates.
    """

    def __init__(self,
                 feature_dimension: int,
                 depth: int = 2,
                 data_map_func: Callable[[np.ndarray], float] = self_product,
                 insert_barriers: bool = False) -> None:
        """
        Args:
            feature_dimension: The number of features
            depth: The number of repeated circuits. Defaults to 2, has a minimum value of 1.
            data_map_func: A mapping function for data x which can be supplied to override the
                default mapping from :meth:`self_product`.
            insert_barriers: If True, barriers are inserted in between the evolution instructions
                and hadamard layers.
        """
        super().__init__(feature_dimension=feature_dimension,
                         depth=depth,
                         z_order=1,
                         data_map_func=data_map_func)
