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

from .pauli_feature_map import PauliFeatureMap


class ZFeatureMap(PauliFeatureMap):
    """The first order Pauli Z-evolution circuit.

    On 3 qubits and with 2 repetitions the circuit is represented by:

    .. parsed-literal::

        ┌───┐┌──────────────┐┌───┐┌──────────────┐
        ┤ H ├┤ U1(2.0*x[0]) ├┤ H ├┤ U1(2.0*x[0]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[1]) ├┤ H ├┤ U1(2.0*x[1]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[2]) ├┤ H ├┤ U1(2.0*x[2]) ├
        └───┘└──────────────┘└───┘└──────────────┘

    This is a sub-class of :class:`~qiskit.circuit.library.PauliFeatureMap` where the Pauli
    strings are fixed as `['Z']`. As a result the first order expansion will be a circuit without
    entangling gates.

    Examples:

        >>> prep = ZFeatureMap(3, reps=3, insert_barriers=True)
        >>> print(prep)
             ┌───┐ ░ ┌──────────────┐ ░ ┌───┐ ░ ┌──────────────┐ ░ ┌───┐ ░ ┌──────────────┐
        q_0: ┤ H ├─░─┤ U1(2.0*x[0]) ├─░─┤ H ├─░─┤ U1(2.0*x[0]) ├─░─┤ H ├─░─┤ U1(2.0*x[0]) ├
             ├───┤ ░ ├──────────────┤ ░ ├───┤ ░ ├──────────────┤ ░ ├───┤ ░ ├──────────────┤
        q_1: ┤ H ├─░─┤ U1(2.0*x[1]) ├─░─┤ H ├─░─┤ U1(2.0*x[1]) ├─░─┤ H ├─░─┤ U1(2.0*x[1]) ├
             ├───┤ ░ ├──────────────┤ ░ ├───┤ ░ ├──────────────┤ ░ ├───┤ ░ ├──────────────┤
        q_2: ┤ H ├─░─┤ U1(2.0*x[2]) ├─░─┤ H ├─░─┤ U1(2.0*x[2]) ├─░─┤ H ├─░─┤ U1(2.0*x[2]) ├
             └───┘ ░ └──────────────┘ ░ └───┘ ░ └──────────────┘ ░ └───┘ ░ └──────────────┘

        >>> data_map = lambda x: x[0]*x[0] + 1  # note: input is an array
        >>> prep = ZFeatureMap(3, reps=1, data_map_func=data_map)
        >>> print(prep)
             ┌───┐┌───────────────────────┐
        q_0: ┤ H ├┤ U1(2.0*x[0]**2 + 2.0) ├
             ├───┤├───────────────────────┤
        q_1: ┤ H ├┤ U1(2.0*x[1]**2 + 2.0) ├
             ├───┤├───────────────────────┤
        q_2: ┤ H ├┤ U1(2.0*x[2]**2 + 2.0) ├
             └───┘└───────────────────────┘

        >>> classifier = ZFeatureMap(3, reps=1) + RY(3, reps=1)
        >>> print(classifier)
             ┌───┐┌──────────────┐┌──────────┐      ┌──────────┐
        q_0: ┤ H ├┤ U1(2.0*x[0]) ├┤ RY(θ[0]) ├─■──■─┤ RY(θ[3]) ├────────────
             ├───┤├──────────────┤├──────────┤ │  │ └──────────┘┌──────────┐
        q_1: ┤ H ├┤ U1(2.0*x[1]) ├┤ RY(θ[1]) ├─■──┼──────■──────┤ RY(θ[4]) ├
             ├───┤├──────────────┤├──────────┤    │      │      ├──────────┤
        q_2: ┤ H ├┤ U1(2.0*x[2]) ├┤ RY(θ[2]) ├────■──────■──────┤ RY(θ[5]) ├
             └───┘└──────────────┘└──────────┘                  └──────────┘

    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        data_map_func: Optional[Callable[[np.ndarray], float]] = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "ZFeatureMap",
    ) -> None:
        """Create a new first-order Pauli-Z expansion circuit.

        Args:
            feature_dimension: The number of features
            reps: The number of repeated circuits. Defaults to 2, has a minimum value of 1.
            data_map_func: A mapping function for data x which can be supplied to override the
                default mapping from :meth:`self_product`.
            parameter_prefix: The prefix used if default parameters are generated.
            insert_barriers: If True, barriers are inserted in between the evolution instructions
                and hadamard layers.

        """
        super().__init__(
            feature_dimension=feature_dimension,
            paulis=["Z"],
            reps=reps,
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            name=name,
        )
