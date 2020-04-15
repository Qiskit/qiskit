# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The two-local gate circuit.

TODO
    * remove the temporary param substitution fix and move to circuits away from gates
    * if entanglement is not a callable, store only 2 blocks, not all of them
    * let identify gate return a type if possible to avoid substitution, handle the circuit
        case differently
"""

from typing import Union, Optional, List, Callable

from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Instruction, Parameter
import qiskit.extensions.standard as gates

from .n_local import NLocal

# disable check for overriding getter and setter because of pylint bug
# pylint: disable=no-member
# pylint:disable=invalid-overridden-method


class TwoLocal(NLocal):
    """The two-local gate circuit.

    TODO
    """

    def __init__(self,
                 num_qubits: Optional[int] = None,
                 rotation_blocks: Optional[Union[str, List[str], type, List[type]]] = None,
                 entanglement_blocks: Optional[Union[str, List[str], type, List[type]]] = None,
                 entanglement: Union[str, List[List[int]], Callable[[int], List[int]]] = 'full',
                 reps: int = 3,
                 skip_unentangled_qubits: bool = False,
                 skip_final_rotation_layer: bool = False,
                 parameter_prefix: str = 'θ',
                 insert_barriers: bool = False,
                 initial_state: Optional['InitialState'] = None,
                 ) -> None:
        """Initializer. Assumes that the type hints are obeyed for now.

        Args:
            num_qubits: The number of qubits of the Ansatz.
            rotation_blocks: The gates used in the rotation layer. Can be specified via the name of
                a gate (e.g. 'ry') or the gate type itself (e.g. RYGate).
                If only one gate is provided, the gate same gate is applied to each qubit.
                If a list of gates is provided, all gates are applied to each qubit in the provided
                order.
                See the Examples section for more detail.
            entanglement_blocks: The gates used in the entanglement layer. Can be specified in
                the same format as `rotation_blocks`.
            entanglement: Specifies the entanglement structure. Can be a string ('full', 'linear'
                or 'sca'), a list of integer-pairs specifying the indices of qubits
                entangled with one another, or a callable returning such a list provided with
                the index of the entanglement layer.
                Default to 'full' entanglement.
                See the Examples section for more detail.
            reps: Specifies how often a block of consisting of a rotation layer and entanglement
                layer is repeated.
            skip_unentangled_qubits: If True, the single qubit gates are only applied to qubits
                that are entangled with another qubit. If False, the single qubit gates are applied
                to each qubit in the Ansatz. Defaults to False.
            skip_final_rotation_layer: If True, a rotation layer is added at the end of the
                ansatz. If False, no rotation layer is added. Defaults to True.
            parameter_prefix: The parameterized gates require a parameter to be defined, for which
                we use instances of `qiskit.circuit.Parameter`. The name of each parameter is the
                number of its occurrence with this specified prefix.
            insert_barriers: If True, barriers are inserted in between each layer. If False,
                no barriers are inserted. Defaults to False.
            initial_state: An `'InitialState'` object to prepend to the Ansatz.
                TODO deprecate this feature in favor of prepend or overloading __add__ in
                the initial state class

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
                 └──────────┘            └──────────┘
        """
        super().__init__(num_qubits=num_qubits,
                         insert_barriers=insert_barriers, initial_state=initial_state,
                         rotation_blocks=rotation_blocks,
                         entanglement_blocks=entanglement_blocks,
                         entanglement=entanglement,
                         reps=reps,
                         skip_final_rotation_layer=skip_final_rotation_layer,
                         skip_unentangled_qubits=skip_unentangled_qubits,
                         parameter_prefix=parameter_prefix)

    def _convert_to_block(self, layer: Union[str, type, Gate, QuantumCircuit]) -> Instruction:
        """For a layer provided as str (e.g. 'ry') or type (e.g. RYGate) this function returns the
        according layer type along with the number of parameters (e.g. (RYGate, 1)).

        Args:
            layer: The qubit layer.

        Returns:
            The specified layer with the required number of parameters.

        Raises:
            ValueError: The type of `layer` is invalid.
            ValueError: The type of `layer` is str but the name is unknown.
            ValueError: The type of `layer` is type but the layer type is unknown.

        Note:
            Outlook: If layers knew their number of parameters as static property, we could also
            allow custom layer types.
        """
        if isinstance(layer, Instruction):
            return layer

        if hasattr(layer, 'to_gate'):
            return layer.to_gate()

        if hasattr(layer, 'to_instruction'):
            return layer.to_instruction()

        # check the list of valid layers
        # this could be a lot easier if the standard layers would have `name` and `num_params`
        # as static types, which might be something they should have anyways
        theta = Parameter('θ')
        valid_layers = {
            'ch': gates.CHGate(),
            'cx': gates.CXGate(),
            'cy': gates.CYGate(),
            'cz': gates.CZGate(),
            'crx': gates.CRXGate(theta),
            'cry': gates.CRYGate(theta),
            'crz': gates.CRZGate(theta),
            'h': gates.HGate(),
            'i': gates.IGate(),
            'id': gates.IGate(),
            'iden': gates.IGate(),
            'rx': gates.RXGate(theta),
            'rxx': gates.RXXGate(theta),
            'ry': gates.RYGate(theta),
            'ryy': gates.RYYGate(theta),
            'rz': gates.RZGate(theta),
            's': gates.SGate(),
            'sdg': gates.SdgGate(),
            'swap': gates.SwapGate(),
            'x': gates.XGate(),
            'y': gates.YGate(),
            'z': gates.ZGate(),
            't': gates.TGate(),
            'tdg': gates.TdgGate(),
        }

        if isinstance(layer, str):
            # iterate over the layer names and look for the specified layer
            for identifier, standard_gate in valid_layers.items():
                if layer == identifier:
                    return standard_gate
            raise ValueError('Unknown layer name `{}`.'.format(layer))

        if isinstance(layer, type):
            # iterate over the layer types and look for the specified layer
            for _, standard_gate in valid_layers.items():
                if isinstance(standard_gate, layer):
                    return standard_gate
            raise ValueError('Unknown layer type`{}`.'.format(layer))

        raise ValueError('Invalid input type {}. '.format(type(layer))
                         + '`layer` must be a type, str or QuantumCircuit.')
