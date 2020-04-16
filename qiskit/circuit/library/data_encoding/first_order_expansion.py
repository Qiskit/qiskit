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

"""Create a new first-order Pauli-Z expansion circuit."""

from typing import Callable, Optional
import numpy as np

from qiskit.util import deprecate_arguments

from .pauli_expansion import PauliExpansion


class FirstOrderExpansion(PauliExpansion):
    """First Order Expansion feature map.

    This is a sub-class of :class:`PauliZExpansion` where *z_order* is fixed at 1.
    As a result the first order expansion will be a feature map without entangling gates.
    """

    @deprecate_arguments({'depth': 'reps'})
    def __init__(self,
                 feature_dimension: int,
                 reps: int = 2,
                 data_map_func: Optional[Callable[[np.ndarray], float]] = None,
                 insert_barriers: bool = False,
                 depth: Optional[int] = None  # pylint: disable=unused-argument
                 ) -> None:
        """Create a new first-order Pauli-Z expansion circuit.

        Args:
            feature_dimension: The number of features
            reps: The number of repeated circuits. Defaults to 2, has a minimum value of 1.
            data_map_func: A mapping function for data x which can be supplied to override the
                default mapping from :meth:`self_product`.
            insert_barriers: If True, barriers are inserted in between the evolution instructions
                and hadamard layers.
            depth: Deprecated, use ``reps`` instead.
        """
        super().__init__(feature_dimension=feature_dimension,
                         paulis=['Z'],
                         reps=reps,
                         data_map_func=data_map_func,
                         insert_barriers=insert_barriers)
