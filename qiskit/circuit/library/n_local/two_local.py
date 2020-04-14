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

from typing import Union, Optional, List, Tuple, Callable

from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Instruction, Parameter
from qiskit.extensions.standard import (IGate, XGate, YGate, ZGate, HGate, TGate, SGate, TdgGate,
                                        SdgGate, RXGate, RXXGate, RYGate, RYYGate, RZGate, SwapGate,
                                        CXGate, CYGate, CZGate, CHGate, CRXGate, CRYGate, CRZGate)

from qiskit.aqua.utils import get_entangler_map, validate_entangler_map

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
                 reps: int = 3,
                 rotation_gates: Optional[Union[str, List[str], type, List[type]]] = None,
                 entanglement_gates: Optional[Union[str, List[str], type, List[type]]] = None,
                 entanglement: Union[str, List[List[int]], Callable[[int], List[int]]] = 'full',
                 initial_state: Optional['InitialState'] = None,
                 skip_unentangled_qubits: bool = False,
                 skip_final_rotation_layer: bool = False,
                 parameter_prefix: str = 'θ',
                 insert_barriers: bool = False,
                 ) -> None:
        """Initializer. Assumes that the type hints are obeyed for now.

        Args:
            num_qubits: The number of qubits of the Ansatz.
            depth: Specifies how often a block of consisting of a rotation layer and entanglement
                layer is repeated.
            rotation_gates: The gates used in the rotation layer. Can be specified via the name of
                a gate (e.g. 'ry') or the gate type itself (e.g. RYGate).
                If only one gate is provided, the gate same gate is applied to each qubit.
                If a list of gates is provided, all gates are applied to each qubit in the provided
                order.
                See the Examples section for more detail.
            entanglement_gates: The gates used in the entanglement layer. Can be specified in
                the same format as `rotation_gates`.
            entanglement: Specifies the entanglement structure. Can be a string ('full', 'linear'
                or 'sca'), a list of integer-pairs specifying the indices of qubits
                entangled with one another, or a callable returning such a list provided with
                the index of the entanglement layer.
                Default to 'full' entanglement.
                See the Examples section for more detail.
            initial_state: An `'InitialState'` object to prepend to the Ansatz.
                TODO deprecate this feature in favor of prepend or overloading __add__ in
                the initial state class
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

        Examples:
            >>> ansatz = TwoLocal(3, 'ry', 'cx', 'linear', reps=2, insert_barriers=True)
            >>> qc = QuantumCircuit(3)  # create a circuit and append the Ansatz
            >>> qc += ansatz.to_circuit()
            >>> qc.decompose().draw()  # decompose the layers into standard gates
                    ┌────────┐ ░            ░ ┌────────┐ ░            ░ ┌────────┐
            q_0: |0>┤ Ry(θ0) ├─░───■────────░─┤ Ry(θ3) ├─░───■────────░─┤ Ry(θ6) ├
                    ├────────┤ ░ ┌─┴─┐      ░ ├────────┤ ░ ┌─┴─┐      ░ ├────────┤
            q_1: |0>┤ Ry(θ1) ├─░─┤ X ├──■───░─┤ Ry(θ4) ├─░─┤ X ├──■───░─┤ Ry(θ7) ├
                    ├────────┤ ░ └───┘┌─┴─┐ ░ ├────────┤ ░ └───┘┌─┴─┐ ░ ├────────┤
            q_2: |0>┤ Ry(θ2) ├─░──────┤ X ├─░─┤ Ry(θ5) ├─░──────┤ X ├─░─┤ Ry(θ8) ├
                    └────────┘ ░      └───┘ ░ └────────┘ ░      └───┘ ░ └────────┘

            >>> ansatz = TwoLocal(3, ['ry','rz'], 'cz', 'full', reps=1, insert_barriers=True)
            >>> print(ansatz)  # quick way of plotting the Ansatz
                    ┌────────┐┌────────┐ ░           ░  ┌────────┐ ┌────────┐
            q_0: |0>┤ Ry(θ0) ├┤ Rz(θ1) ├─░──■──■─────░──┤ Ry(θ6) ├─┤ Rz(θ7) ├
                    ├────────┤├────────┤ ░  │  │     ░  ├────────┤ ├────────┤
            q_1: |0>┤ Ry(θ2) ├┤ Rz(θ3) ├─░──■──┼──■──░──┤ Ry(θ8) ├─┤ Rz(θ9) ├
                    ├────────┤├────────┤ ░     │  │  ░ ┌┴────────┤┌┴────────┤
            q_2: |0>┤ Ry(θ4) ├┤ Rz(θ5) ├─░─────■──■──░─┤ Ry(θ10) ├┤ Rz(θ11) ├
                    └────────┘└────────┘ ░           ░ └─────────┘└─────────┘

            >>> entangler_map = [[0, 1], [1, 2], [2, 0]]  # circular entanglement for 3 qubits
            >>> ansatz = TwoLocal(3, 'x', 'crx', entangler_map, reps=1)
            >>> print(ansatz)  # note: no barriers inserted this time!
                    ┌───┐                         ┌────────┐┌───┐
            q_0: |0>┤ X ├────■────────────────────┤ Rx(θ2) ├┤ X ├
                    ├───┤┌───┴────┐          ┌───┐└───┬────┘└───┘
            q_1: |0>┤ X ├┤ Rx(θ0) ├────■─────┤ X ├────┼──────────
                    ├───┤└────────┘┌───┴────┐└───┘    │     ┌───┐
            q_2: |0>┤ X ├──────────┤ Rx(θ1) ├─────────■─────┤ X ├
                    └───┘          └────────┘               └───┘

            >>> entangler_map = [[0, 3], [0, 2]]  # entangle the first and last two-way
            >>> ansatz = TwoLocal(4, [], 'cry', entangler_map, reps=1)
            >>> circuit = ansatz.to_circuit() + ansatz.to_circuit()  # add two Ansatzes
            >>> circuit.decompose().draw()  # note, that the parameters are the same!
            q_0: |0>────■─────────■─────────■─────────■─────
                        │         │         │         │
            q_1: |0>────┼─────────┼─────────┼─────────┼─────
                        │     ┌───┴────┐    │     ┌───┴────┐
            q_2: |0>────┼─────┤ Ry(θ1) ├────┼─────┤ Ry(θ1) ├
                    ┌───┴────┐└────────┘┌───┴────┐└────────┘
            q_3: |0>┤ Ry(θ0) ├──────────┤ Ry(θ0) ├──────────
                    └────────┘          └────────┘
        """
        super().__init__(num_qubits=num_qubits,
                         insert_barriers=insert_barriers, initial_state=initial_state,
                         rotation_blocks=rotation_gates,
                         entanglement_blocks=entanglement_gates,
                         entanglement=entanglement,
                         reps=reps,
                         parameter_prefix=parameter_prefix)

        # initialize Ansatz
        # super().__init__(insert_barriers=insert_barriers, initial_state=initial_state)

        # # store arguments needing no pre-processing
        # self._depth = depth
        # self._num_qubits = num_qubits
        # self._entanglement = entanglement
        # self._parameter_prefix = parameter_prefix
        # self._skip_unentangled_qubits = skip_unentangled_qubits
        # self._skip_final_rotation_layer = skip_final_rotation_layer

        # # internal variables
        # self._param_count = 0  # class-internal parameter count
        # self._overwrite_block_parameters = False

        # # handle the single- and two-qubit gate specifications
        # self.rotation_gates = rotation_gates or []
        # self.entanglement_gates = entanglement_gates or []

    # def _get_new_parameters(self, n):
    #     new_parameters = [Parameter('{}{}'.format(self._parameter_prefix, i + self._param_count))
    #                       for i in range(n)]
    #     self._param_count += n
    #     self._base_params += new_parameters
    #     return new_parameters

    # def _get_entanglement_layer(self, block_num: int) -> Gate:
    #     """Get the entangler map for this block.

    #     For some kinds of entanglement (e.g. 'sca') the entangler map is differs in different
    #     blocks, therefore we query the entangler map every time. For constant schemata, such as
    #     'linear', this is slightly inefficient, since the entangler map does not change.
    #     However, the number of times get_entangler_map is called equals to `reps` which usually is
    #     of O(10), and therefore most likely no bottleneck

    #     Args:
    #         block_num: The index of the current block.

    #     Returns:
    #         The entanglement layer as gate.
    #     """

    #     circuit = QuantumCircuit(self._num_qubits, name='ent{}'.format(block_num))

    #     for control, target in self.get_entangler_map(block_num):
    #         # apply the gates
    #         for gate, num_params in self.entanglement_gates:
    #             if num_params == 0:
    #                 params = []
    #                 circuit.append(gate, [control, target], params)
    #             else:
    #                 # param_count = self.num_parameters + len(circuit.parameters)
    #                 # params = [Parameter('{}{}'.format(self._parameter_prefix, param_count + i))
    #                 #           for i in range(num_params)]
    #                 # params = [Parameter('{}'.format(param_count + i)) for i in range(num_params)]
    #                 # param_count += num_params
    #                 params = self._get_new_parameters(num_params)

    #                 # correctly replace the parameters
    #                 sub_circuit = QuantumCircuit(self._num_qubits)
    #                 sub_circuit.append(gate, [control, target], [])
    #                 update = dict(zip(list(sub_circuit.parameters), params))
    #                 sub_circuit._substitute_parameters(update)

    #                 # add the gate
    #                 circuit.extend(sub_circuit)

    #     return circuit.to_gate()

    # def _get_rotation_layer(self, block_num: int) -> Gate:
    #     """Get the rotation layer for the current block.

    #     Args:
    #         block_num: The index of the current block.

    #     Returns:
    #         The rotation layer as Gate.
    #     """
    #     # determine the entangled qubits for this block
    #     if self._skip_unentangled_qubits:
    #         all_qubits = []
    #         for control, target in self.get_entangler_map(block_num):
    #             all_qubits.extend([control, target])
    #         entangled_qubits = sorted(list(set(all_qubits)))
    #     else:
    #         entangled_qubits = list(range(self._num_qubits))

    #     # build the circuit for this block
    #     circuit = QuantumCircuit(self._num_qubits, name='rot{}'.format(block_num))

    #     # iterate over all qubits
    #     for qubit in range(self._num_qubits):

    #         # check if we need to apply the gate to the qubit
    #         if not self._skip_unentangled_qubits or qubit in entangled_qubits:

    #             # apply the gates
    #             for gate, num_params in self.rotation_gates:
    #                 if num_params == 0:
    #                     params = []
    #                     circuit.append(gate, [qubit], params)  # TODO use _append with register
    #                 else:
    #                     params = self._get_new_parameters(num_params)

    #                     # correctly replace the parameters
    #                     sub_circuit = QuantumCircuit(self._num_qubits)
    #                     sub_circuit.append(gate, [qubit], [])
    #                     update = dict(zip(list(sub_circuit.parameters), params))
    #                     sub_circuit._substitute_parameters(update)

    #                     # add the gate
    #                     circuit.extend(sub_circuit)

    #     return circuit.to_gate()

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
            'ch': (CHGate(), 0),
            'cx': (CXGate(), 0),
            'cy': (CYGate(), 0),
            'cz': (CZGate(), 0),
            'crx': (CRXGate(theta), 1),
            'cry': (CRYGate(theta), 1),
            'crz': (CRZGate(theta), 1),
            'h': (HGate(), 0),
            'i': (IGate(), 0),
            'id': (IGate(), 0),
            'iden': (IGate(), 0),
            'rx': (RXGate(theta), 1),
            'rxx': (RXXGate(theta), 1),
            'ry': (RYGate(theta), 1),
            'ryy': (RYYGate(theta), 1),
            'rz': (RZGate(theta), 1),
            's': (SGate(), 0),
            'sdg': (SdgGate(), 0),
            'swap': (SwapGate(), 0),
            'x': (XGate(), 0),
            'y': (YGate(), 0),
            'z': (ZGate(), 0),
            't': (TGate(), 0),
            'tdg': (TdgGate(), 0),
        }

        if isinstance(layer, str):
            # iterate over the layer names and look for the specified layer
            for identifier, (standard_gate, _) in valid_layers.items():
                if layer == identifier:
                    return standard_gate
            raise ValueError('Unknown layer name `{}`.'.format(layer))

        if isinstance(layer, type):
            # iterate over the layer types and look for the specified layer
            for _, (standard_gate, _) in valid_layers.items():
                if isinstance(standard_gate, layer):
                    return standard_gate
            raise ValueError('Unknown layer type`{}`.'.format(layer))

        raise ValueError('Invalid input type {}. '.format(type(layer))
                         + '`layer` must be a type, str or QuantumCircuit.')

    # @NLocal.blocks.getter
    # def blocks(self) -> List[Instruction]:
    #     """Set the blocks according to the current state.

    #     Set the blocks and return them.
    #     """
    #     if self._blocks:
    #         return self._blocks

    #     if self._num_qubits is None:
    #         raise ValueError('The number of qubits has not been set!')

    #     if self.rotation_gates is None:
    #         raise ValueError('No rotation gates are specified.')

    #     if self.entanglement_gates is None:
    #         raise ValueError('No entanglement gates are specified.')

    #     blocks = []
    #     self._param_count = 0
    #     self._base_params = []
    #     # define the blocks of this NLocal
    #     for block_num in range(self._depth):
    #         # append a rotation layer, if entanglement gates are specified
    #         if len(self._rotation_gates) > 0:
    #             block = self._get_rotation_layer(block_num)
    #             blocks += [block]

    #         # append an entanglement layer, if entanglement gates are specified
    #         if len(self._entanglement_gates) > 0:
    #             block = self._get_entanglement_layer(block_num)
    #             blocks += [block]

    #     # add a final rotation layer, if not specified otherwise
    #     if not self._skip_final_rotation_layer and len(self._rotation_gates) > 0:
    #         block = self._get_rotation_layer(block_num)
    #         blocks += [block]

    #     self._blocks = blocks
    #     return blocks

    # @NLocal.base_parameters.getter
    # def base_parameters(self) -> List[Parameter]:
    #     """Return the parameters of the underlying circuit.

    #     Returns:
    #         The parameters used in the circuit.
    #     """
    #     _ = self.blocks
    #     return self._base_params

    # @NLocal.num_qubits.setter
    # def num_qubits(self, num_qubits: int) -> None:
    #     """Set the number of qubits.

    #     Args:
    #         num_qubits: The new number of qubits.

    #     Note:
    #         Additionally to invalidating the circuit (which is done in NLocal.num_qubits),
    #         here we need to invalidate the blocks, since they are dependent on the number of qubits.
    #     """
    #     if num_qubits != self._num_qubits:
    #         self._blocks, self._circuit = None, None  # invalidate current setup
    #         self._num_qubits = num_qubits

    # @property
    # def depth(self) -> int:
    #     """Return the depth of the two-local NLocal.

    #     The depth specifies how often the sequence of rotation and entanglement layer is
    #     repeated. Additionally, a rotation layer is appended at the end (which can be
    #     turned off using the attribute `skip_final_rotation_layer`).

    #     Example:
    #         >>> ansatz = TwoLocal(2, 2, 'ry', 'cx')  # the second argument, depth, is 2
    #         >>> ansatz
    #                 ┌────────┐     ┌────────┐     ┌────────┐
    #         q_0: |0>┤ Ry(θ0) ├──■──┤ Ry(θ2) ├──■──┤ Ry(θ4) ├
    #                 ├────────┤┌─┴─┐├────────┤┌─┴─┐├────────┤
    #         q_1: |0>┤ Ry(θ1) ├┤ X ├┤ Ry(θ3) ├┤ X ├┤ Ry(θ5) ├
    #                 └────────┘└───┘└────────┘└───┘└────────┘
    #         >>> ansatz.depth = 1
    #         >>> ansatz
    #                 ┌────────┐     ┌────────┐
    #         q_0: |0>┤ Ry(θ0) ├──■──┤ Ry(θ2) ├
    #                 ├────────┤┌─┴─┐├────────┤
    #         q_1: |0>┤ Ry(θ1) ├┤ X ├┤ Ry(θ3) ├
    #                 └────────┘└───┘└────────┘

    #     Returns:
    #         The depth.
    #     """
    #     return self._depth

    # @depth.setter
    # def depth(self, depth: int) -> None:
    #     """Set the new depth.

    #     Args:
    #         depth: The new depth.
    #     """
    #     if depth != self._depth:
    #         self._blocks, self._circuit = None, None
    #         self._depth = depth

    # @property
    # def entanglement(self) -> Union[str, List[List[int]], callable]:
    #     """Return the current entanglement strategy.

    #     Can be of type ``str``, a list of ``int`` or a ``callable``.
    #     If the entanglement is a ``str`` it can be one of:
    #     - ``'full'``: Entangle each qubit with every qubit (all-to-all).
    #     - ``'linear'``: Entangle each qubit with its direct neighbor.
    #     - ``'sca'``: Shifted-circular-alternating entanglement. Within every entanglement block,
    #     the qubits are entangled circularly, i.e. linear but the first and last qubit are
    #     also entangled. Shifted means that in the next block the (i+1)-th entanglement is the
    #     i-th entanglement of the previous block (modulo ``num_qubits``). Alternating indicates
    #     that the role of target and control qubit is swapped every other block.

    #     If the entanglement is a list of lists of ``int``, the structure is specified as
    #     ``[[control1, target1], [control2, target2], ...]``.

    #     If the entanglement is a callable, it takes the number of the current entanglement block
    #     as argument and returns the entanglement list for that particular block, i.e. returns
    #     a list of lists of ``int``.

    #     Returns:
    #         The entanglement strategy.
    #     """
    #     return self._entanglement

    # @entanglement.setter
    # def entanglement(self, entanglement: Union[str, List[List[int]], callable]):
    #     """Set the entanglement strategy.

    #     Args:
    #         entanglement: The entanglement strategy.
    #     """
    #     self._blocks, self._circuit = None, None  # invalidate current setup
    #     self._entanglement = entanglement

    # @property
    # def entanglement_gates(self) -> List[Tuple[type, int]]:
    #     """
    #     Return a the twos qubit gate(or gates) in form of callable(s).

    #     Returns:
    #         list[tuple]: the single qubit gate(s) as tuples (QuantumCircuit.gate, num_parameters),
    #             e.g. (QuantumCircuit.cx, 0) or (QuantumCircuit.cry, 1)
    #     """
    #     gate_param_list = [TwoLocal.identify_gate(gate) for gate in self._entanglement_gates]
    #     return gate_param_list

    # @entanglement_gates.setter
    # def entanglement_gates(self, gates):
    #     """Set new entanglement gates."""
    #     # invalidate circuit definition
    #     self._blocks, self._circuit = None, None

    #     if not isinstance(gates, list):
    #         self._entanglement_gates = [gates]
    #     else:
    #         self._entanglement_gates = gates

    # @property
    # def rotation_gates(self) -> List[Tuple[type, int]]:
    #     """Return a the single qubit gate (or gates) in tuples of callable and number of parameters.

    #     The reason this is implemented as separate function is that the user can set up a class
    #     with special single and two qubit gates, for cases we do not cover in identify gate.
    #     And this design "outsources" the identification of the gate from the main code that
    #     builds the circuit, which makes the code more modular.

    #     Returns:
    #         list[tuple]: the single qubit gate(s) as tuples (QuantumCircuit.gate, num_parameters),
    #             e.g. (QuantumCircuit.x, 0) or (QuantumCircuit.ry, 1)
    #     """
    #     gate_param_list = [TwoLocal.identify_gate(gate) for gate in self._rotation_gates]
    #     return gate_param_list

    # @rotation_gates.setter
    # def rotation_gates(self, gates):
    #     """Set new rotation gates."""
    #     # invalidate circuit definition
    #     self._blocks, self._circuit = None, None

    #     if not isinstance(gates, list):
    #         self._rotation_gates = [gates]
    #     else:
    #         self._rotation_gates = gates

    # def get_entangler_map(self, offset: int = 0) -> List[List[int]]:
    #     """Return the specified entangler map, if self._entangler_map if it has been set previously.

    #     Args:
    #         offset (int): Some entanglements allow an offset argument, since the entangler map might
    #             differ per entanglement block (e.g. for 'sca' entanglement). This is the block
    #             index.

    #     Returns:
    #         A list of [control, target] pairs specifying entanglements, also known as entangler map.

    #     Raises:
    #         ValueError: Unsupported format of entanglement, if self._entanglement has the wrong
    #             format.
    #     """
    #     if isinstance(self._entanglement, str):
    #         return get_entangler_map(self._entanglement, self._num_qubits, offset)
    #     elif callable(self._entanglement):
    #         return validate_entangler_map(self._entanglement(offset), self._num_qubits)
    #     elif isinstance(self._entanglement, list):
    #         return validate_entangler_map(self._entanglement, self._num_qubits)
    #     else:
    #         raise ValueError('Unsupported format of entanglement!')
