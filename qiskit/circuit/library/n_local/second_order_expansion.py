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
"""
This module contains the definition of a base class for
feature map. Several types of commonly used approaches.
"""

from typing import Callable, List, Union, Optional
import numpy as np
from qiskit.util import deprecate_arguments
from .pauli_z_expansion import PauliZExpansion
from .data_mapping import self_product


class SecondOrderExpansion(PauliZExpansion):
    """Second Order Expansion feature map.

    This is a sub-class of :class:`PauliZExpansion` where *z_order* is fixed at 2.
    """

    @deprecate_arguments({'entangler_map': 'entanglement'})
    def __init__(self,
                 feature_dimension: int,
                 depth: int = 2,
                 entanglement: Union[str, List[List[int]], Callable[[int], List[int]]] = 'full',
                 data_map_func: Callable[[np.ndarray], float] = self_product,
                 insert_barriers: bool = False,
                 entangler_map: Optional[List[List[int]]] = None  # pylint:disable=unused-argument
                 ) -> None:
        """

        Args:
            feature_dimension: Number of features.
            depth: The number of repeated circuits, has a min. value of 1.
            entanglement: Specifies the entanglement structure. Can be a string ('full', 'linear'
                or 'sca'), a list of integer-pairs specifying the indices of qubits
                entangled with one another, or a callable returning such a list provided with
                the index of the entanglement layer.
                Default to 'full' entanglement.
            data_map_func: A mapping function for data x.
            insert_barriers: If True, barriers are inserted in between the evolution instructions
                and hadamard layers.
            entangler_map: Deprecated, use ``entanglement`` instead.
        """
        super().__init__(feature_dimension=feature_dimension,
                         depth=depth,
                         entanglement=entanglement,
                         z_order=2,
                         data_map_func=data_map_func)
