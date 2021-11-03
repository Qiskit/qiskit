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
from .pauli_feature_map import PauliFeatureMap


class ZZFeatureMap(PauliFeatureMap):
    """Second-order Pauli-Z evolution circuit.

    For 3 qubits and 1 repetition and linear entanglement the circuit is represented by:

    .. parsed-literal::

        ┌───┐┌─────────────────┐
        ┤ H ├┤ U1(2.0*φ(x[0])) ├──■────────────────────────────■────────────────────────────────────
        ├───┤├─────────────────┤┌─┴─┐┌──────────────────────┐┌─┴─┐
        ┤ H ├┤ U1(2.0*φ(x[1])) ├┤ X ├┤ U1(2.0*φ(x[0],x[1])) ├┤ X ├──■────────────────────────────■──
        ├───┤├─────────────────┤└───┘└──────────────────────┘└───┘┌─┴─┐┌──────────────────────┐┌─┴─┐
        ┤ H ├┤ U1(2.0*φ(x[2])) ├──────────────────────────────────┤ X ├┤ U1(2.0*φ(x[1],x[2])) ├┤ X ├
        └───┘└─────────────────┘                                  └───┘└──────────────────────┘└───┘

    where ``φ`` is a classical non-linear function, which defaults to ``φ(x) = x`` if and
    ``φ(x,y) = (pi - x)(pi - y)``.

    Examples:

        >>> prep = ZZFeatureMap(2, reps=2)
        >>> print(prep)
             ┌───┐┌──────────────┐
        q_0: ┤ H ├┤ U1(2.0*x[0]) ├──■───────────────────────────────────────■──
             ├───┤├──────────────┤┌─┴─┐┌─────────────────────────────────┐┌─┴─┐
        q_1: ┤ H ├┤ U1(2.0*x[1]) ├┤ X ├┤ U1(2.0*(pi - x[0])*(pi - x[1])) ├┤ X ├
             └───┘└──────────────┘└───┘└─────────────────────────────────┘└───┘

        >>> from qiskit.circuit.library import EfficientSU2
        >>> classifier = ZZFeatureMap(3) + EfficientSU2(3)
        >>> classifier.num_parameters
        15
        >>> classifier.parameters  # 'x' for the data preparation, 'θ' for the SU2 parameters
        {Parameter(θ[9]), Parameter(θ[4]), Parameter(θ[6]), Parameter(θ[1]), Parameter(x[2]),
        Parameter(θ[7]), Parameter(x[1]), Parameter(θ[8]), Parameter(θ[2]), Parameter(θ[10]),
        Parameter(θ[5]), Parameter(θ[0]), Parameter(θ[3]), Parameter(x[0]), Parameter(θ[11])}
        >>> classifier.count_ops()
        OrderedDict([('u1', 12), ('cx', 12), ('ry', 12), ('cz', 9), ('h', 6)])
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: Union[str, List[List[int]], Callable[[int], List[int]]] = "full",
        data_map_func: Optional[Callable[[np.ndarray], float]] = None,
        insert_barriers: bool = False,
        name: str = "ZZFeatureMap",
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

        Raises:
            ValueError: If the feature dimension is smaller than 2.
        """
        if feature_dimension < 2:
            raise ValueError(
                "The ZZFeatureMap contains 2-local interactions and cannot be "
                f"defined for less than 2 qubits. You provided {feature_dimension}."
            )

        super().__init__(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            paulis=["Z", "ZZ"],
            data_map_func=data_map_func,
            insert_barriers=insert_barriers,
            name=name,
        )
