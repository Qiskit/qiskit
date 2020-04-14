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

from typing import Optional, Callable, List, Union
import numpy as np
from .pauli_expansion import PauliExpansion
from .data_mapping import self_product


class PauliZExpansion(PauliExpansion):
    """The Pauli Z Expansion feature map.

    This is a sub-class of the general :class:`PauliExpansion` but where the pauli string is fixed
    to only contain Z and where *paulis* is now created for the superclass as per the given
    *z_order*. So with default of 2 this creates ['Z', 'ZZ'] which also happens to be the default
    of the superclass. A *z_order* of 3 would be ['Z', 'ZZ', 'ZZZ'] and so on.
    """

    def __init__(self,
                 feature_dimension: int,
                 depth: int = 2,
                 entanglement: Union[str, List[List[int]], Callable[[int], List[int]]] = 'full',
                 paulis: Optional[List[str]] = None,
                 z_order: int = 2,
                 data_map_func: Callable[[np.ndarray], float] = self_product,
                 insert_barriers: bool = False) -> None:
        """
        Args:
            feature_dimension: Number of features.
            depth: The number of repeated circuits. Defaults to 2, has a min. value of 1.
            entanglement: Specifies the entanglement structure. Can be a string ('full', 'linear'
                or 'sca'), a list of integer-pairs specifying the indices of qubits
                entangled with one another, or a callable returning such a list provided with
                the index of the entanglement layer.
                Default to 'full' entanglement.
            paulis: A list of strings for to-be-used paulis. Defaults to None.
                If None, ['Z', 'ZZ'] will be used.
            z_order: z order.
            data_map_func: A mapping function for data x which can be supplied to override the
                default mapping from :meth:`self_product`.
            insert_barriers: If True, barriers are inserted in between the evolution instructions
                and hadamard layers.
        """
        pauli_string = []
        for i in range(1, z_order + 1):
            pauli_string.append('Z' * i)

        super().__init__(feature_dimension=feature_dimension,
                         depth=depth,
                         entanglement=entanglement,
                         paulis=paulis,
                         data_map_func=data_map_func,
                         insert_barriers=insert_barriers)
