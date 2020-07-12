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

"""The two-local gate circuit."""

from typing import Union, Optional, List, Callable, Any

from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Instruction, Parameter

from .n_local import NLocal
from ..standard_gates import (
    IGate, XGate, YGate, ZGate, RXGate, RYGate, RZGate, HGate, SGate, SdgGate, TGate, TdgGate,
    RXXGate, RYYGate, RZXGate, RZZGate, SwapGate, CXGate, CYGate, CZGate, CRXGate, CRYGate, CRZGate,
    CHGate
)


class TwoLocal(NLocal):
    r"""The two-local circuit.

    The two-local circuit is a parameterized circuit consisting of alternating rotation layers and
    entanglement layers. The rotation layers are single qubit gates applied on all qubits.
    The entanglement layer uses two-qubit gates to entangle the qubits according to a strategy set
    using ``entanglement``. Both the rotation and entanglement gates can be specified as
    string (e.g. ``'ry'`` or ``'cx'``), as gate-type (e.g. ``RYGate`` or ``CXGate``) or
    as QuantumCircuit (e.g. a 1-qubit circuit or 2-qubit circuit).

    A set of default entanglement strategies is provided:

    * ``'full'`` entanglement is each qubit is entangled with all the others.
    * ``'linear'`` entanglement is qubit :math:`i` entangled with qubit :math:`i + 1`,
      for all :math:`i \in \{0, 1, ... , n - 2\}`, where :math:`n` is the total number of qubits.
    * ``'circular'`` entanglement is linear entanglement but with an additional entanglement of the
      first and last qubit before the linear part.
    * ``'sca'`` (shifted-circular-alternating) entanglement is a generalized and modified version
      of the proposed circuit 14 in `Sim et al. <https://arxiv.org/abs/1905.10876>`__.
      It consists of circular entanglement where the 'long' entanglement connecting the first with
      the last qubit is shifted by one each block.  Furthermore the role of control and target
      qubits are swapped every block (therefore alternating).

    The entanglement can further be specified using an entangler map, which is a list of index
    pairs, such as

    >>> entangler_map = [(0, 1), (1, 2), (2, 0)]

    If different entanglements per block should be used, provide a list of entangler maps.
    See the examples below on how this can be used.

    >>> entanglement = [entangler_map_layer_1, entangler_map_layer_2, ... ]

    Barriers can be inserted in between the different layers for better visualization using the
    ``insert_barriers`` attribute.

    For each parameterized gate a new parameter is generated using a
    :class:`~qiskit.circuit.library.ParameterVector`. The name of these parameters can be chosen
    using the ``parameter_prefix``.

    Examples:

        >>> two = TwoLocal(3, 'ry', 'cx', 'linear', reps=2, insert_barriers=True)
        >>> print(two)  # decompose the layers into standard gates
             ┌──────────┐ ░            ░ ┌──────────┐ ░            ░ ┌──────────┐
        q_0: ┤ Ry(θ[0]) ├─░───■────────░─┤ Ry(θ[3]) ├─░───■────────░─┤ Ry(θ[6]) ├
             ├──────────┤ ░ ┌─┴─┐      ░ ├──────────┤ ░ ┌─┴─┐      ░ ├──────────┤
        q_1: ┤ Ry(θ[1]) ├─░─┤ X ├──■───░─┤ Ry(θ[4]) ├─░─┤ X ├──■───░─┤ Ry(θ[7]) ├
             ├──────────┤ ░ └───┘┌─┴─┐ ░ ├──────────┤ ░ └───┘┌─┴─┐ ░ ├──────────┤
        q_2: ┤ Ry(θ[2]) ├─░──────┤ X ├─░─┤ Ry(θ[5]) ├─░──────┤ X ├─░─┤ Ry(θ[8]) ├
             └──────────┘ ░      └───┘ ░ └──────────┘ ░      └───┘ ░ └──────────┘

        >>> two = TwoLocal(3, ['ry','rz'], 'cz', 'full', reps=1, insert_barriers=True)
        >>> qc = QuantumCircuit(3)
        >>> qc += two
        >>> print(qc.decompose().draw())
             ┌──────────┐┌──────────┐ ░           ░ ┌──────────┐ ┌──────────┐
        q_0: ┤ Ry(θ[0]) ├┤ Rz(θ[3]) ├─░──■──■─────░─┤ Ry(θ[6]) ├─┤ Rz(θ[9]) ├
             ├──────────┤├──────────┤ ░  │  │     ░ ├──────────┤┌┴──────────┤
        q_1: ┤ Ry(θ[1]) ├┤ Rz(θ[4]) ├─░──■──┼──■──░─┤ Ry(θ[7]) ├┤ Rz(θ[10]) ├
             ├──────────┤├──────────┤ ░     │  │  ░ ├──────────┤├───────────┤
        q_2: ┤ Ry(θ[2]) ├┤ Rz(θ[5]) ├─░─────■──■──░─┤ Ry(θ[8]) ├┤ Rz(θ[11]) ├
             └──────────┘└──────────┘ ░           ░ └──────────┘└───────────┘

        >>> entangler_map = [[0, 1], [1, 2], [2, 0]]  # circular entanglement for 3 qubits
        >>> two = TwoLocal(3, 'x', 'crx', entangler_map, reps=1)
        >>> print(two)  # note: no barriers inserted this time!
                ┌───┐                             ┌──────────┐┌───┐
        q_0: |0>┤ X ├─────■───────────────────────┤ Rx(θ[2]) ├┤ X ├
                ├───┤┌────┴─────┐            ┌───┐└─────┬────┘└───┘
        q_1: |0>┤ X ├┤ Rx(θ[0]) ├─────■──────┤ X ├──────┼──────────
                ├───┤└──────────┘┌────┴─────┐└───┘      │     ┌───┐
        q_2: |0>┤ X ├────────────┤ Rx(θ[1]) ├───────────■─────┤ X ├
                └───┘            └──────────┘                 └───┘

        >>> entangler_map = [[0, 3], [0, 2]]  # entangle the first and last two-way
        >>> two = TwoLocal(4, [], 'cry', entangler_map, reps=1)
        >>> circuit = two + two
        >>> print(circuit.decompose().draw())  # note, that the parameters are the same!
        q_0: ─────■───────────■───────────■───────────■──────
                  │           │           │           │
        q_1: ─────┼───────────┼───────────┼───────────┼──────
                  │      ┌────┴─────┐     │      ┌────┴─────┐
        q_2: ─────┼──────┤ Ry(θ[1]) ├─────┼──────┤ Ry(θ[1]) ├
             ┌────┴─────┐└──────────┘┌────┴─────┐└──────────┘
        q_3: ┤ Ry(θ[0]) ├────────────┤ Ry(θ[0]) ├────────────
             └──────────┘            └─────────

        >>> layer_1 = [(0, 1), (0, 2)]
        >>> layer_2 = [(1, 2)]
        >>> two = TwoLocal(3, 'x', 'cx', [layer_1, layer_2], reps=2, insert_barriers=True)
        >>> print(two)
             ┌───┐ ░            ░ ┌───┐ ░       ░ ┌───┐
        q_0: ┤ X ├─░───■────■───░─┤ X ├─░───────░─┤ X ├
             ├───┤ ░ ┌─┴─┐  │   ░ ├───┤ ░       ░ ├───┤
        q_1: ┤ X ├─░─┤ X ├──┼───░─┤ X ├─░───■───░─┤ X ├
             ├───┤ ░ └───┘┌─┴─┐ ░ ├───┤ ░ ┌─┴─┐ ░ ├───┤
        q_2: ┤ X ├─░──────┤ X ├─░─┤ X ├─░─┤ X ├─░─┤ X ├
             └───┘ ░      └───┘ ░ └───┘ ░ └───┘ ░ └───┘

    """

    def __init__(self,
                 num_qubits: Optional[int] = None,
                 rotation_blocks: Optional[Union[str, List[str], type, List[type],
                                                 QuantumCircuit, List[QuantumCircuit]]] = None,
                 entanglement_blocks: Optional[Union[str, List[str], type, List[type],
                                                     QuantumCircuit, List[QuantumCircuit]]] = None,
                 entanglement: Union[str, List[List[int]], Callable[[int], List[int]]] = 'full',
                 reps: int = 3,
                 skip_unentangled_qubits: bool = False,
                 skip_final_rotation_layer: bool = False,
                 parameter_prefix: str = 'θ',
                 insert_barriers: bool = False,
                 initial_state: Optional[Any] = None,
                 ) -> None:
        """Construct a new two-local circuit.

        Args:
            num_qubits: The number of qubits of the two-local circuit.
            rotation_blocks: The gates used in the rotation layer. Can be specified via the name of
                a gate (e.g. 'ry') or the gate type itself (e.g. RYGate).
                If only one gate is provided, the gate same gate is applied to each qubit.
                If a list of gates is provided, all gates are applied to each qubit in the provided
                order.
                See the Examples section for more detail.
            entanglement_blocks: The gates used in the entanglement layer. Can be specified in
                the same format as `rotation_blocks`.
            entanglement: Specifies the entanglement structure. Can be a string ('full', 'linear'
                , 'circular' or 'sca'), a list of integer-pairs specifying the indices of qubits
                entangled with one another, or a callable returning such a list provided with
                the index of the entanglement layer.
                Default to 'full' entanglement.
                See the Examples section for more detail.
            reps: Specifies how often a block consisting of a rotation layer and entanglement
                layer is repeated.
            skip_unentangled_qubits: If True, the single qubit gates are only applied to qubits
                that are entangled with another qubit. If False, the single qubit gates are applied
                to each qubit in the Ansatz. Defaults to False.
            skip_final_rotation_layer: If False, a rotation layer is added at the end of the
                ansatz. If True, no rotation layer is added.
            parameter_prefix: The parameterized gates require a parameter to be defined, for which
                we use instances of `qiskit.circuit.Parameter`. The name of each parameter will
                be this specified prefix plus its index.
            insert_barriers: If True, barriers are inserted in between each layer. If False,
                no barriers are inserted. Defaults to False.
            initial_state: An `InitialState` object to prepend to the circuit.

        """
        super().__init__(num_qubits=num_qubits,
                         rotation_blocks=rotation_blocks,
                         entanglement_blocks=entanglement_blocks,
                         entanglement=entanglement,
                         reps=reps,
                         skip_final_rotation_layer=skip_final_rotation_layer,
                         skip_unentangled_qubits=skip_unentangled_qubits,
                         insert_barriers=insert_barriers,
                         initial_state=initial_state,
                         parameter_prefix=parameter_prefix)

    def _convert_to_block(self, layer: Union[str, type, Gate, QuantumCircuit]) -> QuantumCircuit:
        """For a layer provided as str (e.g. 'ry') or type (e.g. RYGate) this function returns the
        according layer type along with the number of parameters (e.g. (RYGate, 1)).

        Args:
            layer: The qubit layer.

        Returns:
            The specified layer with the required number of parameters.

        Raises:
            TypeError: The type of `layer` is invalid.
            ValueError: The type of `layer` is str but the name is unknown.
            ValueError: The type of `layer` is type but the layer type is unknown.

        Note:
            Outlook: If layers knew their number of parameters as static property, we could also
            allow custom layer types.
        """
        if isinstance(layer, QuantumCircuit):
            return layer

        # check the list of valid layers
        # this could be a lot easier if the standard layers would have `name` and `num_params`
        # as static types, which might be something they should have anyways
        theta = Parameter('θ')
        valid_layers = {
            'ch': CHGate(),
            'cx': CXGate(),
            'cy': CYGate(),
            'cz': CZGate(),
            'crx': CRXGate(theta),
            'cry': CRYGate(theta),
            'crz': CRZGate(theta),
            'h': HGate(),
            'i': IGate(),
            'id': IGate(),
            'iden': IGate(),
            'rx': RXGate(theta),
            'rxx': RXXGate(theta),
            'ry': RYGate(theta),
            'ryy': RYYGate(theta),
            'rz': RZGate(theta),
            'rzx': RZXGate(theta),
            'rzz': RZZGate(theta),
            's': SGate(),
            'sdg': SdgGate(),
            'swap': SwapGate(),
            'x': XGate(),
            'y': YGate(),
            'z': ZGate(),
            't': TGate(),
            'tdg': TdgGate(),
        }

        # try to exchange `layer` from a string to a gate instance
        if isinstance(layer, str):
            try:
                layer = valid_layers[layer]
            except KeyError:
                raise ValueError('Unknown layer name `{}`.'.format(layer))

        # try to exchange `layer` from a type to a gate instance
        if isinstance(layer, type):
            # iterate over the layer types and look for the specified layer
            instance = None
            for gate in valid_layers.values():
                if isinstance(gate, layer):
                    instance = gate
            if instance is None:
                raise ValueError('Unknown layer type`{}`.'.format(layer))
            layer = instance

        if isinstance(layer, Instruction):
            circuit = QuantumCircuit(layer.num_qubits)
            circuit.append(layer, list(range(layer.num_qubits)))
            return circuit

        raise TypeError('Invalid input type {}. '.format(type(layer))
                        + '`layer` must be a type, str or QuantumCircuit.')

    def get_entangler_map(self, rep_num: int, block_num: int, num_block_qubits: int
                          ) -> List[List[int]]:
        """Overloading to handle the special case of 1 qubit where the entanglement are ignored."""
        if self.num_qubits <= 1:
            return []
        return super().get_entangler_map(rep_num, block_num, num_block_qubits)
