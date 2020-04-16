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
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.extensions.standard import HGate
from qiskit.util import deprecate_arguments

from .data_mapping import self_product
from .n_local import NLocal


class PauliExpansion(NLocal):
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
            \prod_{j \in S} (\pi - x_j)

    Please refer to :class:`FirstOrderExpansion` for the case :math:`k = 1`, :math:`P_0 = Z`
    and to :class:`SecondOrderExpansion` for the case :math:`k = 2`, :math:`P_0 = Z` and
    :math:`P_1 P_0 = ZZ`.

    References:
        [1]: Havlicek et al. (2018), Supervised learning with quantum enhanced feature spaces.
            https://arxiv.org/abs/1804.11326.
    """

    @deprecate_arguments({'depth': 'reps'})
    def __init__(self,
                 feature_dimension: int,
                 reps: int = 2,
                 entanglement: Union[str, List[List[int]], Callable[[int], List[int]]] = 'full',
                 paulis: Optional[List[str]] = None,
                 data_map_func: Callable[[np.ndarray], float] = self_product,
                 insert_barriers: bool = False,
                 depth: Optional[int] = None,  # pylint: disable=unused-argument
                 ) -> None:
        """
        Args:
            feature_dimension: Number of qubits in the circuit.
            reps: The number of repeated circuits.
            entanglement: Specifies the entanglement structure. Refer to
                :class:`~qiskit.circuit.library.NLocal` for detail.
            paulis: A list of strings for to-be-used paulis. If None are provided, ``['Z', 'ZZ']``
                will be used.
            data_map_func: A mapping function for data x which can be supplied to override the
                default mapping from :meth:`self_product`.
            insert_barriers: If True, barriers are inserted in between the evolution instructions
                and hadamard layers.
            depth: Deprecated, use ``reps`` instead.
        """

        super().__init__(num_qubits=feature_dimension,
                         reps=reps,
                         rotation_blocks=HGate(),
                         entanglement_blocks=[],
                         entanglement=entanglement,
                         insert_barriers=insert_barriers)

        self._data_map_func = data_map_func

        paulis = paulis or ['Z', 'ZZ']
        self.entanglement_blocks = [self.pauli_block(pauli) for pauli in paulis]

    @property
    def feature_dimension(self) -> int:
        """Returns the feature dimension (which is equal to the number of qubits).

        Returns:
            The feature dimension of this feature map.
        """
        return self.num_qubits

    def _extract_data_for_rotation(self, pauli, x):
        where_non_i = np.where(np.asarray(list(pauli[::-1])) != 'I')[0]
        x = np.asarray(x)
        return x[where_non_i]

    def pauli_block(self, pauli_string):
        """Get the Pauli block for the feature map circuit."""
        params = ParameterVector('x', length=self.feature_dimension)
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
        evo.u1(2 * time, indices[-1])
        cx_chain(evo, inverse=True)
        basis_change(evo, inverse=True)
        return evo
