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

import itertools

from typing import Optional, Callable, List, Union, Sequence
from functools import reduce
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library.standard_gates import HGate

from ..n_local.n_local import NLocal


class PauliFeatureMap(NLocal):
    r"""The Pauli Expansion circuit.

    The Pauli Expansion circuit is a data encoding circuit that transforms input data
    :math:`\vec{x} \in \mathbb{R}^n`, where `n` is the ``feature_dimension``, as

    .. math::

        U_{\Phi(\vec{x})}=\exp\left(i\sum_{S \in \mathcal{I}}
        \phi_S(\vec{x})\prod_{i\in S} P_i\right).

    Here, :math:`S` is a set of qubit indices that describes the connections in the feature map,
    :math:`\mathcal{I}` is a set containing all these index sets, and
    :math:`P_i \in \{I, X, Y, Z\}`. Per default the data-mapping
    :math:`\phi_S` is

    .. math::

        \phi_S(\vec{x}) = \begin{cases}
            x_i \text{ if } S = \{i\} \\
            \prod_{j \in S} (\pi - x_j) \text{ if } |S| > 1
            \end{cases}.

    The possible connections can be set using the ``entanglement`` and ``paulis`` arguments.
    For example, for single-qubit :math:`Z` rotations and two-qubit :math:`YY` interactions
    between all qubit pairs, we can set::


        feature_map = PauliFeatureMap(..., paulis=["Z", "YY"], entanglement="full")

    which will produce blocks of the form

    .. parsed-literal::

        ┌───┐┌──────────────┐┌──────────┐                                             ┌───────────┐
        ┤ H ├┤ U1(2.0*x[0]) ├┤ RX(pi/2) ├──■───────────────────────────────────────■──┤ RX(-pi/2) ├
        ├───┤├──────────────┤├──────────┤┌─┴─┐┌─────────────────────────────────┐┌─┴─┐├───────────┤
        ┤ H ├┤ U1(2.0*x[1]) ├┤ RX(pi/2) ├┤ X ├┤ U1(2.0*(pi - x[0])*(pi - x[1])) ├┤ X ├┤ RX(-pi/2) ├
        └───┘└──────────────┘└──────────┘└───┘└─────────────────────────────────┘└───┘└───────────┘

    The circuit contains ``reps`` repetitions of this transformation.

    Please refer to :class:`.ZFeatureMap` for the case of single-qubit Pauli-:math:`Z` rotations
    and to :class:`.ZZFeatureMap` for the single- and two-qubit Pauli-:math:`Z` rotations.

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
        >>> classifier = prep.compose(wavefunction)
        >>> classifier.num_parameters
        27
        >>> classifier.count_ops()
        OrderedDict([('cx', 39), ('rx', 36), ('u1', 21), ('h', 15), ('ry', 12), ('rz', 12)])

    References:



    [1] Havlicek et al. Supervised learning with quantum enhanced feature spaces,
    `Nature 567, 209-212 (2019) <https://www.nature.com/articles/s41586-019-0980-2>`__.

    """

    def __init__(
        self,
        feature_dimension: Optional[int] = None,
        reps: int = 2,
        entanglement: Union[str, List[List[int]], Callable[[int], List[int]]] = "full",
        alpha: float = 2.0,
        paulis: Optional[List[str]] = None,
        data_map_func: Optional[Callable[[np.ndarray], float]] = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "PauliFeatureMap",
    ) -> None:
        """Create a new Pauli expansion circuit.

        Args:
            feature_dimension: Number of qubits in the circuit.
            reps: The number of repeated circuits.
            entanglement: Specifies the entanglement structure. Refer to
                :class:`~qiskit.circuit.library.NLocal` for detail.
            alpha: The Pauli rotation factor, multiplicative to the pauli rotations
            paulis: A list of strings for to-be-used paulis. If None are provided, ``['Z', 'ZZ']``
                will be used.
            data_map_func: A mapping function for data x which can be supplied to override the
                default mapping from :meth:`self_product`.
            parameter_prefix: The prefix used if default parameters are generated.
            insert_barriers: If True, barriers are inserted in between the evolution instructions
                and hadamard layers.

        """

        super().__init__(
            num_qubits=feature_dimension,
            reps=reps,
            rotation_blocks=HGate(),
            entanglement=entanglement,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            skip_final_rotation_layer=True,
            name=name,
        )

        self._data_map_func = data_map_func or self_product
        self._paulis = paulis or ["Z", "ZZ"]
        self._alpha = alpha
        self.entanglement = entanglement

    def _parameter_generator(
        self, rep: int, block: int, indices: List[int]
    ) -> Optional[List[Parameter]]:
        """If certain blocks should use certain parameters this method can be overridden."""
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
    def alpha(self) -> float:
        """The Pauli rotation factor (alpha).

        Returns:
            The Pauli rotation factor.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        """Set the Pauli rotation factor (alpha).

        Args:
            alpha: Pauli rotation factor
        """
        self._invalidate()
        self._alpha = alpha

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
        where_non_i = np.where(np.asarray(list(pauli[::-1])) != "I")[0]
        x = np.asarray(x)
        return x[where_non_i]

    def pauli_block(self, pauli_string):
        """Get the Pauli block for the feature map circuit."""
        params = ParameterVector("_", length=len(pauli_string))
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
            if pauli != "I":
                trimmed += [pauli]
                indices += [i]

        evo = QuantumCircuit(len(pauli_string))

        if len(trimmed) == 0:
            return evo

        def basis_change(circuit, inverse=False):
            for i, pauli in enumerate(pauli_string):
                if pauli == "X":
                    circuit.h(i)
                elif pauli == "Y":
                    circuit.rx(-np.pi / 2 if inverse else np.pi / 2, i)

        def cx_chain(circuit, inverse=False):
            num_cx = len(indices) - 1
            for i in reversed(range(num_cx)) if inverse else range(num_cx):
                circuit.cx(indices[i], indices[i + 1])

        basis_change(evo)
        cx_chain(evo)
        evo.p(self.alpha * time, indices[-1])
        cx_chain(evo, inverse=True)
        basis_change(evo, inverse=True)
        return evo

    # helper function to generate correct entangler map
    def _selective_entangler_map(self, num_block_qubits, entanglement):
        # if entanglement is not specified for a pauli then use 'full' entanglement
        if all(num_block_qubits != len(ent_block) for ent_block in entanglement):
            entanglement_map = list(
                itertools.combinations(list(range(self.feature_dimension)), num_block_qubits)
            )
            return entanglement_map
        # if any entanglement is specified then it will only be used for
        # the correct size of block, like for entanglement=[(0,1), (1,2,3)]
        # only [(0,1)] will be used by 2-qubit pauli (like 'ZZ') and [(1,2,3)]
        # will be used by 3-qubit pauli (like 'ZZZ')
        else:
            entanglement_map = [ent for ent in entanglement if len(ent) == num_block_qubits]
            return entanglement_map

    def get_entangler_map(
        self, rep_num: int, block_num: int, num_block_qubits: int
    ) -> Sequence[Sequence[int]]:
        i, j = rep_num, block_num
        num_i = len(self.entanglement)
        num_j = len(self.entanglement[i % num_i])

        # entanglement is List[List[int]]
        if all(isinstance(e2, (int, np.int32, np.int64)) for en in self.entanglement for e2 in en):
            for ind, en in enumerate(self.entanglement):
                if not any(len(en) == len(pauli) for pauli in self.paulis):
                    raise ValueError(f"Invalid value of entanglement:{en}")
                self.entanglement[ind] = tuple(map(int, en))
            return self._selective_entangler_map(num_block_qubits, self.entanglement)

        # entanglement is List[List[List[int]]]
        elif all(
            isinstance(e3, (int, np.int32, np.int64))
            for en in self.entanglement
            for e2 in en
            for e3 in e2
        ):
            for en in self.entanglement:
                for ind, e2 in enumerate(en):
                    if not any(len(e2) == len(pauli) for pauli in self.paulis):
                        raise ValueError(f"Invalid value of entanglement:{e2}")
                    en[ind] = tuple(map(int, e2))

            # choose the entanglement based on the reps
            chosen_entanglement = self.entanglement[i % num_i]
            return self._selective_entangler_map(num_block_qubits, chosen_entanglement)

        # entanglement is List[List[List[List[int]]]]
        elif all(
            isinstance(e4, (int, np.int32, np.int64))
            for en in self.entanglement
            for e2 in en
            for e3 in e2
            for e4 in e3
        ):
            for en in self.entanglement:
                for e2 in en:
                    for ind, e3 in enumerate(e2):
                        if not any(len(e3) == len(pauli) for pauli in self.paulis):
                            raise ValueError(f"Invalid value of entanglement:{e3}")
                        e2[ind] = tuple(map(int, e3))

            # Unlike Twolocal where we can specify entanglement blocks and rotation
            # blocks separately, for PauliFeatureMap, all the paulis are specified
            # as a single list (like ['Z', 'ZZ', 'YY']) and so if we just use
            # self.entanglement[i % num_i][j % num_j] as the entanglement we will be
            # choosing incorrect entanglement from the specified entanglement. So,
            # here we subtract the number of single-qubit paulis from the j % num_j
            # to pick correct entanglement from the specified List[List[List[List[int]]]]
            count_single_qubit_paulis = 0
            for pauli in self.paulis:
                if len(pauli) == 1:
                    count_single_qubit_paulis += 1

            chosen_entanglement = self.entanglement[i % num_i][
                (j % num_j) - count_single_qubit_paulis
            ]
            return self._selective_entangler_map(num_block_qubits, chosen_entanglement)

        else:
            # if the entanglement is not List[List[int]] or List[List[List[int]]] or
            # List[List[List[List[int]]]] then we fall back on the original
            # `get_entangler_map()` method from NLocal
            return super().get_entangler_map(rep_num, block_num, num_block_qubits)


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
