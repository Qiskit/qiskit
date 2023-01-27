# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Free-Axis Selection (Fraxis) circuit."""

from typing import Any, Callable, List, Optional, Union

from qiskit.circuit import Gate, Instruction, Parameter
from qiskit.circuit.quantumcircuit import QuantumCircuit

from ..standard_gates import (
    CHGate,
    CXGate,
    CYGate,
    CZGate,
    HGate,
    IGate,
    SdgGate,
    SGate,
    SwapGate,
    TdgGate,
    TGate,
    U3Gate,
    UGate,
    XGate,
    YGate,
    ZGate,
)
from .n_local import NLocal


class FraxisCircuit(NLocal):
    r"""Free-Axis Selection (Fraxis) circuit.

    The Fraxis circuit is a parameterized circuit consisting of alternating rotation layers and
    entanglement layers. The rotation layers are single qubit U3 gates applied on all qubits
    The entanglement layer uses two-qubit gates to entangle the qubits according to a strategy set
    using ``entanglement``. Both the rotation and entanglement gates can be specified as
    string (e.g. ``'u'`` or ``'cx'``), as gate-type (e.g. ``UGate`` or ``CXGate``) or
    as QuantumCircuit (e.g. a 1-qubit circuit or 2-qubit circuit).

    The Fraxis circuit is supposed to be optimized with
    :class:`~qiskit.algorithms.optimizers.FraxisOptimizer`.
    See `HC. Watanabe et al. <https://arxiv.org/abs/2104.14875>`__ for details.

    Note 1: Only U(U3) gate is allowed for rotation layers. Other single qubit gates with parameters
    such as Ry and Rz gates are not allowed.

    Note 2: Only entanglement gates without parameters are allowed for entanglement layers.
    Entanglement gates with parameters such as CRx and Rxx gates are not allowed.

    A set of default entanglement strategies is provided:

    * ``'full'`` entanglement is each qubit is entangled with all the others.
    * ``'linear'`` entanglement is qubit :math:`i` entangled with qubit :math:`i + 1`,
      for all :math:`i \in \{0, 1, ... , n - 2\}`, where :math:`n` is the total number of qubits.
    * ``'reverse_linear'`` entanglement is qubit :math:`i` entangled with qubit :math:`i + 1`,
      for all :math:`i \in \{n-2, n-3, ... , 1, 0\}`, where :math:`n` is the total number of qubits.
      Note that if ``entanglement_blocks = 'cx'`` then this option provides the same unitary as
      ``'full'`` with fewer entangling gates.
    * ``'pairwise'`` entanglement is one layer where qubit :math:`i` is entangled with qubit
      :math:`i + 1`, for all even values of :math:`i`, and then a second layer where qubit :math:`i`
      is entangled with qubit :math:`i + 1`, for all odd values of :math:`i`.
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

        >>> circ = FraxisCircuit(3, 'u', 'cx', 'linear', reps=2, insert_barriers=True)
        >>> print(circ.decompose())
             ┌───────────────────┐ ░            ░ ┌─────────────────────┐  ░            ░ ┌──────────────────────┐
        q_0: ┤ U(θ[0],θ[1],θ[2]) ├─░───■────────░─┤ U(θ[9],θ[10],θ[11]) ├──░───■────────░─┤ U(θ[18],θ[19],θ[20]) ├
             ├───────────────────┤ ░ ┌─┴─┐      ░ ├─────────────────────┴┐ ░ ┌─┴─┐      ░ ├──────────────────────┤
        q_1: ┤ U(θ[3],θ[4],θ[5]) ├─░─┤ X ├──■───░─┤ U(θ[12],θ[13],θ[14]) ├─░─┤ X ├──■───░─┤ U(θ[21],θ[22],θ[23]) ├
             ├───────────────────┤ ░ └───┘┌─┴─┐ ░ ├──────────────────────┤ ░ └───┘┌─┴─┐ ░ ├──────────────────────┤
        q_2: ┤ U(θ[6],θ[7],θ[8]) ├─░──────┤ X ├─░─┤ U(θ[15],θ[16],θ[17]) ├─░──────┤ X ├─░─┤ U(θ[24],θ[25],θ[26]) ├
             └───────────────────┘ ░      └───┘ ░ └──────────────────────┘ ░      └───┘ ░ └──────────────────────┘

        >>> circ = FraxisCircuit(3, ['h','u'], 'cz', 'full', reps=1, insert_barriers=True)
        >>> print(circ.decompose())
             ┌───┐┌───────────────────┐ ░           ░ ┌───┐┌─────────────────────┐
        q_0: ┤ H ├┤ U(θ[0],θ[1],θ[2]) ├─░──■──■─────░─┤ H ├┤ U(θ[9],θ[10],θ[11]) ├─
             ├───┤├───────────────────┤ ░  │  │     ░ ├───┤├─────────────────────┴┐
        q_1: ┤ H ├┤ U(θ[3],θ[4],θ[5]) ├─░──■──┼──■──░─┤ H ├┤ U(θ[12],θ[13],θ[14]) ├
             ├───┤├───────────────────┤ ░     │  │  ░ ├───┤├──────────────────────┤
        q_2: ┤ H ├┤ U(θ[6],θ[7],θ[8]) ├─░─────■──■──░─┤ H ├┤ U(θ[15],θ[16],θ[17]) ├
             └───┘└───────────────────┘ ░           ░ └───┘└──────────────────────┘

        >>> entangler_map = [[0, 1], [1, 2], [2, 0]]  # circular entanglement for 3 qubits
        >>> circ = FraxisCircuit(3, 'u', 'cx', entangler_map, reps=1)
        >>> print(circ.decompose())  # note: no barriers inserted this time!
             ┌───────────────────┐                                  ┌───┐┌─────────────────────┐
        q_0: ┤ U(θ[0],θ[1],θ[2]) ├──■───────────────────────────────┤ X ├┤ U(θ[9],θ[10],θ[11]) ├─
             ├───────────────────┤┌─┴─┐     ┌──────────────────────┐└─┬─┘└─────────────────────┘
        q_1: ┤ U(θ[3],θ[4],θ[5]) ├┤ X ├──■──┤ U(θ[12],θ[13],θ[14]) ├──┼──────────────────────────
             ├───────────────────┤└───┘┌─┴─┐└──────────────────────┘  │  ┌──────────────────────┐
        q_2: ┤ U(θ[6],θ[7],θ[8]) ├─────┤ X ├──────────────────────────■──┤ U(θ[15],θ[16],θ[17]) ├
             └───────────────────┘     └───┘                             └──────────────────────┘

    """

    def __init__(
        self,
        num_qubits: Optional[int] = None,
        rotation_blocks: Union[
            str, List[str], type, List[type], QuantumCircuit, List[QuantumCircuit]
        ] = "u",
        entanglement_blocks: Union[
            str, List[str], type, List[type], QuantumCircuit, List[QuantumCircuit]
        ] = "cx",
        entanglement: Union[str, List[List[int]], Callable[[int], List[int]]] = "pairwise",
        reps: int = 3,
        skip_unentangled_qubits: bool = False,
        skip_final_rotation_layer: bool = False,
        parameter_prefix: str = "θ",
        insert_barriers: bool = False,
        initial_state: Optional[Any] = None,
        name: str = "Fraxis",
    ) -> None:
        """Construct a new Fraxis circuit.

        Args:
            num_qubits: The number of qubits of the two-local circuit.
            rotation_blocks: The gates used in the rotation layer. Can be specified via the name of
                a gate (e.g. ``'ry'``) or the gate type itself (e.g. :class:`.RYGate`).
                If only one gate is provided, the gate same gate is applied to each qubit.
                If a list of gates is provided, all gates are applied to each qubit in the provided
                order.
                See the Examples section for more detail.
            entanglement_blocks: The gates used in the entanglement layer. Can be specified in
                the same format as ``rotation_blocks``.
            entanglement: Specifies the entanglement structure. Can be a string (``'full'``,
                ``'linear'``, ``'pairwise'``, ``'reverse_linear'``, ``'circular'`` or ``'sca'``),
                a list of integer-pairs specifying the indices
                of qubits entangled with one another, or a callable returning such a list provided with
                the index of the entanglement layer.
                Default to ``'pairwise'`` entanglement.
                Note that if ``entanglement_blocks = 'cx'``, then ``'full'`` entanglement provides the
                same unitary as ``'reverse_linear'`` but the latter option has fewer entangling gates.
                See the Examples section for more detail.
            reps: Specifies how often a block consisting of a rotation layer and entanglement
                layer is repeated.
            skip_unentangled_qubits: If ``True``, the single qubit gates are only applied to qubits
                that are entangled with another qubit. If ``False``, the single qubit gates are applied
                to each qubit in the ansatz. Defaults to ``False``.
            skip_final_rotation_layer: If ``False``, a rotation layer is added at the end of the
                ansatz. If ``True``, no rotation layer is added.
            parameter_prefix: The parameterized gates require a parameter to be defined, for which
                we use instances of :class:`~qiskit.circuit.Parameter`. The name of each parameter will
                be this specified prefix plus its index.
            insert_barriers: If ``True``, barriers are inserted in between each layer. If ``False``,
                no barriers are inserted. Defaults to ``False``.
            initial_state: A :class:`.QuantumCircuit` object to prepend to the circuit.

        """
        super().__init__(
            num_qubits=num_qubits,
            rotation_blocks=rotation_blocks,
            entanglement_blocks=entanglement_blocks,
            entanglement=entanglement,
            reps=reps,
            skip_final_rotation_layer=skip_final_rotation_layer,
            skip_unentangled_qubits=skip_unentangled_qubits,
            insert_barriers=insert_barriers,
            initial_state=initial_state,
            parameter_prefix=parameter_prefix,
            name=name,
        )

    def _convert_to_block(self, layer: Union[str, type, Gate, QuantumCircuit]) -> QuantumCircuit:
        """For a layer provided as str (e.g. ``'u'``) or type (e.g. :class:`.UGate`) this function
         returns the
         according layer type along with the number of parameters (e.g. ``(UGate, 3)``).

        Args:
            layer: The qubit layer.

        Returns:
            The specified layer with the required number of parameters.

        Raises:
            TypeError: The type of ``layer`` is invalid.
            ValueError: The type of ``layer`` is str but the name is unknown.
            ValueError: The type of ``layer`` is type but the layer type is unknown.

        Note:
            Outlook: If layers knew their number of parameters as static property, we could also
            allow custom layer types.
        """
        if isinstance(layer, QuantumCircuit):
            return layer

        theta = Parameter("θ")
        phi = Parameter("θ2")
        lam = Parameter("θ3")
        valid_layers = {
            "u": UGate(theta, phi, lam),
            "u3": U3Gate(theta, phi, lam),
            "ch": CHGate(),
            "cx": CXGate(),
            "cy": CYGate(),
            "cz": CZGate(),
            "h": HGate(),
            "i": IGate(),
            "id": IGate(),
            "iden": IGate(),
            "s": SGate(),
            "sdg": SdgGate(),
            "swap": SwapGate(),
            "x": XGate(),
            "y": YGate(),
            "z": ZGate(),
            "t": TGate(),
            "tdg": TdgGate(),
        }

        # try to exchange `layer` from a string to a gate instance
        if isinstance(layer, str):
            try:
                layer = valid_layers[layer]
            except KeyError as ex:
                raise ValueError(f"Unknown layer name `{layer}`.") from ex

        # try to exchange `layer` from a type to a gate instance
        if isinstance(layer, type):
            # iterate over the layer types and look for the specified layer
            instance = None
            for gate in valid_layers.values():
                if isinstance(gate, layer):
                    instance = gate
            if instance is None:
                raise ValueError(f"Unknown layer type`{layer}`.")
            layer = instance

        if isinstance(layer, Instruction):
            circuit = QuantumCircuit(layer.num_qubits)
            circuit.append(layer, list(range(layer.num_qubits)))
            return circuit

        raise TypeError(
            f"Invalid input type {type(layer)}. " + "`layer` must be a type, str or QuantumCircuit."
        )

    def get_entangler_map(
        self, rep_num: int, block_num: int, num_block_qubits: int
    ) -> List[List[int]]:
        """Overloading to handle the special case of 1 qubit where the entanglement are ignored."""
        if self.num_qubits <= 1:
            return []
        return super().get_entangler_map(rep_num, block_num, num_block_qubits)
