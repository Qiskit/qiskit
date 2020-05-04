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

"""The Pauli expansion circuit module."""

from typing import Optional, Callable, List, Union
from functools import reduce
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library.standard_gates import HGate

from ..n_local.n_local import NLocal


class PauliFeatureMap(NLocal):
    r"""The Pauli Expansion circuit.

    The Pauli Expansion circuit is a data encoding circuit that transforms input data
    :math:`\vec{x} \in \mathbb{R}^n` as

    .. math::

        U_{\Phi(\vec{x})}=\exp\left(i\sum_{S\subseteq [n]}
        \phi_S(\vec{x})\prod_{i\in S} P_i\right)

    The circuit contains ``reps`` repetitions of this transformation.
    The variable :math:`P_i \in \{ I, X, Y, Z \}` denotes the Pauli matrices.
    The index :math:`S` describes connectivities between different qubits or datapoints:
    :math:`S \in \{\binom{n}{k}\ combinations,\ k = 1,... n \}`. Per default the data-mapping
    :math:`\phi_S` is

    .. math::

        \phi_S(\vec{x}) = \begin{cases}
            x_0 \text{ if } k = 1 \\
            \prod_{j \in S} (\pi - x_j) \text{ otherwise }
            \end{cases}

    For example, if the Pauli strings are chosen to be :math:`P_0 = Z` and :math:`P_{0,1} = YY` on
    2 qubits and with 1 repetition using the default data-mapping, the Pauli evolution feature map
    is represented by:

    .. parsed-literal::

        ┌───┐┌──────────────┐┌──────────┐                                             ┌───────────┐
        ┤ H ├┤ U1(2.0*x[0]) ├┤ RX(pi/2) ├──■───────────────────────────────────────■──┤ RX(-pi/2) ├
        ├───┤├──────────────┤├──────────┤┌─┴─┐┌─────────────────────────────────┐┌─┴─┐├───────────┤
        ┤ H ├┤ U1(2.0*x[1]) ├┤ RX(pi/2) ├┤ X ├┤ U1(2.0*(pi - x[0])*(pi - x[1])) ├┤ X ├┤ RX(-pi/2) ├
        └───┘└──────────────┘└──────────┘└───┘└─────────────────────────────────┘└───┘└───────────┘

    Please refer to :class:`ZFeatureMap` for the case :math:`k = 1`, :math:`P_0 = Z`
    and to :class:`ZZFeatureMap` for the case :math:`k = 2`, :math:`P_0 = Z` and
    :math:`P_{0,1} = ZZ`.

    Examples:

        >>> prep = PauliFeatureMap(2, reps=1, paulis=['ZZ'])
        >>> print(prep)
             ┌───┐
        q_0: ┤ H ├──■───────────────────────────────────────■──
             ├───┤┌─┴─┐┌─────────────────────────────────┐┌─┴─┐
        q_1: ┤ H ├┤ X ├┤ U1(2.0*(pi - x[0])*(pi - x[1])) ├┤ X ├
             └───┘└───┘└─────────────────────────────────┘└───┘

        >>> prep = PauliFeatureMap(2, reps=1, paulis=['Z', 'XX'])
        >>> print(prep)
             ┌───┐┌──────────────┐┌───┐                                             ┌───┐
        q_0: ┤ H ├┤ U1(2.0*x[0]) ├┤ H ├──■───────────────────────────────────────■──┤ H ├
             ├───┤├──────────────┤├───┤┌─┴─┐┌─────────────────────────────────┐┌─┴─┐├───┤
        q_1: ┤ H ├┤ U1(2.0*x[1]) ├┤ H ├┤ X ├┤ U1(2.0*(pi - x[0])*(pi - x[1])) ├┤ X ├┤ H ├
             └───┘└──────────────┘└───┘└───┘└─────────────────────────────────┘└───┘└───┘

        >>> prep = PauliFeatureMap(2, reps=1, paulis=['ZY'])
        >>> print(prep)
             ┌───┐┌──────────┐                                             ┌───────────┐
        q_0: ┤ H ├┤ RX(pi/2) ├──■───────────────────────────────────────■──┤ RX(-pi/2) ├
             ├───┤└──────────┘┌─┴─┐┌─────────────────────────────────┐┌─┴─┐└───────────┘
        q_1: ┤ H ├────────────┤ X ├┤ U1(2.0*(pi - x[0])*(pi - x[1])) ├┤ X ├─────────────
             └───┘            └───┘└─────────────────────────────────┘└───┘

        >>> from qiskit.circuit.library import EfficientSU2
        >>> prep = PauliFeatureMap(3, reps=3, paulis=['Z', 'YY', 'ZXZ'])
        >>> wavefunction = EfficientSU2(3)
        >>> classifier = prep.compose(wavefunction
        >>> classifier.num_parameters
        27
        >>> classifier.count_ops()
        OrderedDict([('cx', 39), ('rx', 36), ('u1', 21), ('h', 15), ('ry', 12), ('rz', 12)])

    References:
        [1]: Havlicek et al. (2018), Supervised learning with quantum enhanced feature spaces.
           `arXiv:1804.11326 <https://arxiv.org/abs/1804.11326>`_
    """

    def __init__(self,
                 feature_dimension: Optional[int] = None,
                 reps: int = 2,
                 entanglement: Union[str, List[List[int]], Callable[[int], List[int]]] = 'full',
                 paulis: Optional[List[str]] = None,
                 data_map_func: Optional[Callable[[np.ndarray], float]] = None,
                 parameter_prefix: str = 'x',
                 insert_barriers: bool = False,
                 ) -> None:
        """Create a new Pauli expansion circuit.

        Args:
            feature_dimension: Number of qubits in the circuit.
            reps: The number of repeated circuits.
            entanglement: Specifies the entanglement structure. Refer to
                :class:`~qiskit.circuit.library.NLocal` for detail.
            paulis: A list of strings for to-be-used paulis. If None are provided, ``['Z', 'ZZ']``
                will be used.
            data_map_func: A mapping function for data x which can be supplied to override the
                default mapping from :meth:`self_product`.
            parameter_prefix: The prefix used if default parameters are generated.
            insert_barriers: If True, barriers are inserted in between the evolution instructions
                and hadamard layers.

        """

        super().__init__(num_qubits=feature_dimension,
                         reps=reps,
                         rotation_blocks=HGate(),
                         entanglement=entanglement,
                         parameter_prefix=parameter_prefix,
                         insert_barriers=insert_barriers,
                         skip_final_rotation_layer=True)

        self._data_map_func = data_map_func or self_product
        self._paulis = paulis or ['Z', 'ZZ']

    # pylint: disable=unused-argument
    def _parameter_generator(self, rep: int, block: int, indices: List[int]
                             ) -> Optional[List[Parameter]]:
        """If certain blocks should use certain parameters this method can be overriden."""
        params = [self.ordered_parameters[i] for i in indices]
        return params

    @property
    def num_parameters_settable(self):
        """The number of distinct parameters."""
        return self.feature_dimension

    @property
    def paulis(self) -> List[str]:
        """The Pauli strings used in the entanglement of the qubits.

        Returns:
            The Pauli strings as list.
        """
        return self._paulis

    @paulis.setter
    def paulis(self, paulis: List[str]) -> None:
        """Set the pauli strings.

        Args:
            paulis: The new pauli strings.
        """
        self._invalidate()
        self._paulis = paulis

    @property
    def entanglement_blocks(self):
        return [self.pauli_block(pauli) for pauli in self._paulis]

    @entanglement_blocks.setter
    def entanglement_blocks(self, entanglement_blocks):
        self._entanglement_blocks = entanglement_blocks

    @property
    def feature_dimension(self) -> int:
        """Returns the feature dimension (which is equal to the number of qubits).

        Returns:
            The feature dimension of this feature map.
        """
        return self.num_qubits

    @feature_dimension.setter
    def feature_dimension(self, feature_dimension: int) -> None:
        """Set the feature dimension.

        Args:
            feature_dimension: The new feature dimension.
        """
        self.num_qubits = feature_dimension

    def _extract_data_for_rotation(self, pauli, x):
        where_non_i = np.where(np.asarray(list(pauli[::-1])) != 'I')[0]
        x = np.asarray(x)
        return x[where_non_i]

    def pauli_block(self, pauli_string):
        """Get the Pauli block for the feature map circuit."""
        params = ParameterVector('_', length=len(pauli_string))
        time = self._data_map_func(np.asarray(params))
        return self.pauli_evolution(pauli_string, time)

    def pauli_evolution(self, pauli_string, time):
        """Get the evolution block for the given pauli string."""
        # for some reason this is in reversed order
        pauli_string = pauli_string[::-1]

        # trim the pauli string if identities are included
        trimmed = []
        indices = []
        for i, pauli in enumerate(pauli_string):
            if pauli != 'I':
                trimmed += [pauli]
                indices += [i]

        evo = QuantumCircuit(len(pauli_string))

        if len(trimmed) == 0:
            return evo

        def basis_change(circuit, inverse=False):
            for i, pauli in enumerate(pauli_string):
                if pauli == 'X':
                    circuit.h(i)
                elif pauli == 'Y':
                    circuit.rx(-np.pi / 2 if inverse else np.pi / 2, i)

        def cx_chain(circuit, inverse=False):
            num_cx = len(indices) - 1
            for i in reversed(range(num_cx)) if inverse else range(num_cx):
                circuit.cx(indices[i], indices[i + 1])

        basis_change(evo)
        cx_chain(evo)
        evo.u1(2.0 * time, indices[-1])
        cx_chain(evo, inverse=True)
        basis_change(evo, inverse=True)
        return evo


def self_product(x: np.ndarray) -> float:
    """
    Define a function map from R^n to R.

    Args:
        x: data

    Returns:
        float: the mapped value
    """
    coeff = x[0] if len(x) == 1 else reduce(lambda m, n: m * n, np.pi - x)
    return coeff
