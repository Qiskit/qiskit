# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The tensor product ansatz."""

from typing import Union, Optional, List, Any, Callable

import numpy
from qiskit.circuit import QuantumCircuit, Gate, Instruction, Parameter

from .n_local import NLocal, get_entangler_map
from ..standard_gates import (
    IGate,
    XGate,
    YGate,
    ZGate,
    RXGate,
    RYGate,
    RZGate,
    HGate,
    SGate,
    SdgGate,
    TGate,
    TdgGate,
    RXXGate,
    RYYGate,
    RZXGate,
    RZZGate,
    SwapGate,
    CXGate,
    CYGate,
    CZGate,
    CRXGate,
    CRYGate,
    CRZGate,
    CHGate,
)


class TensorProductAnsatz(NLocal):
    r"""The tensor product ansatz."""

    def __init__(
        self,
        num_qubits: int,
        block_size: int,
        rotation_blocks: Optional[
            Union[str, List[str], type, List[type], QuantumCircuit, List[QuantumCircuit]]
        ] = "ry",
        entanglement_blocks: Optional[
            Union[str, List[str], type, List[type], QuantumCircuit, List[QuantumCircuit]]
        ] = "cx",
        entanglement: Union[str, List[List[int]], Callable[[int], List[int]]] = "linear",
        reps: int = 3,
        insert_barriers: bool = False,
        parameter_prefix: str = "θ",
        skip_final_rotation_layer: Optional[bool] = False,
        initial_state: Optional[Any] = None,
        name: Optional[str] = "TensorProductAnsatz",
    ) -> None:
        """
        Args:
            num_qubits:
            block_size:
            rotation_blocks:
            entanglement_blocks:
            entanglement:
            reps:
            insert_barriers:
            parameter_prefix:
            skip_final_rotation_layer:
            initial_state:
            name:
        """
        self.block_size = block_size

        super().__init__(
            num_qubits=num_qubits,
            rotation_blocks=rotation_blocks,
            entanglement_blocks=entanglement_blocks,
            entanglement=entanglement,
            reps=reps,
            insert_barriers=insert_barriers,
            parameter_prefix=parameter_prefix,
            skip_final_rotation_layer=skip_final_rotation_layer,
            initial_state=initial_state,
            name=name,
        )

    def _build_entanglement_layer(self, circuit, param_iter, i):
        """Build an entanglement layer."""
        # iterate over all entanglement blocks
        for j, block in enumerate(self.entanglement_blocks):
            # create a new layer
            layer = QuantumCircuit(*self.qregs)

            # get entangler map for small block and create it
            entangler_map = self.get_entangler_map(i, j, block.num_qubits, self.block_size)
            small_block = QuantumCircuit(self.block_size)
            for indices in entangler_map:
                parameterized_block = self._parameterize_block(block, param_iter, i, j, indices)
                small_block.compose(parameterized_block, indices, inplace=True)

            # we apply the entanglement gates stacked on top of each other, i.e.
            # if we have 4 qubits and an entanglement block of width 2, we apply two instances
            block_indices = [
                list(range(k * small_block.num_qubits, (k + 1) * small_block.num_qubits))
                for k in range(self.num_qubits // small_block.num_qubits)
            ]

            # apply the operations in the layer
            for indices in block_indices:
                parameterized_block = self._parameterize_block(
                    small_block, param_iter, i, j, indices
                )
                layer.compose(parameterized_block, indices, inplace=True)

            # add the layer to the circuit
            circuit.compose(layer, inplace=True)

    # pylint: disable=too-many-return-statements
    def get_entangler_map(
        self,
        rep_num: int,
        block_num: int,
        num_block_qubits: int,
        num_circuit_qubits: Optional[int] = None,
    ) -> List[List[int]]:
        """Get the entangler map for in the repetition ``rep_num`` and the block ``block_num``.

        The entangler map for the current block is derived from the value of ``self.entanglement``.
        Below the different cases are listed, where ``i`` and ``j`` denote the repetition number
        and the block number, respectively, and ``n`` the number of qubits in the block.

        entanglement type              | entangler map
        -------------------------------+--------------------------------------------------------
        None                           | [[0, ..., n - 1]]
        str (e.g 'full')               | the specified connectivity on ``n`` qubits
        List[int]                      | [``entanglement``]
        List[List[int]]                | ``entanglement``
        List[List[List[int]]]          | ``entanglement[i]``
        List[List[List[List[int]]]]    | ``entanglement[i][j]``
        List[str]                      | the connectivity specified in ``entanglement[i]``
        List[List[str]]                | the connectivity specified in ``entanglement[i][j]``
        Callable[int, str]             | same as List[str]
        Callable[int, List[List[int]]] | same as List[List[List[int]]]

        Note that all indices are to be taken modulo the length of the array they act on, i.e.
        no out-of-bounds index error will be raised but we re-iterate from the beginning of the
        list.

        Args:
            rep_num: The current repetition we are in.
            block_num: The block number within the entanglement layers.
            num_block_qubits: The number of qubits in the block.
            num_circuit_qubits: The number of qubits in the circuit.

        Returns:
            The entangler map for the current block in the current repetition.

        Raises:
            ValueError: If the value of ``entanglement`` could not be cast to a corresponding
                entangler map.
        """
        if num_circuit_qubits is None:
            num_circuit_qubits = self.num_qubits
        i, j, n = rep_num, block_num, num_block_qubits
        entanglement = self._entanglement

        # entanglement is None
        if entanglement is None:
            return [list(range(n))]

        # entanglement is callable
        if callable(entanglement):
            entanglement = entanglement(i)

        # entanglement is str
        if isinstance(entanglement, str):
            return get_entangler_map(n, num_circuit_qubits, entanglement, offset=i)

        # check if entanglement is list of something
        if not isinstance(entanglement, (tuple, list)):
            raise ValueError(f"Invalid value of entanglement: {entanglement}")
        num_i = len(entanglement)

        # entanglement is List[str]
        if all(isinstance(en, str) for en in entanglement):
            return get_entangler_map(n, num_circuit_qubits, entanglement[i % num_i], offset=i)

        # entanglement is List[int]
        if all(isinstance(en, (int, numpy.integer)) for en in entanglement):
            return [[int(en) for en in entanglement]]

        # check if entanglement is List[List]
        if not all(isinstance(en, (tuple, list)) for en in entanglement):
            raise ValueError(f"Invalid value of entanglement: {entanglement}")
        num_j = len(entanglement[i % num_i])

        # entanglement is List[List[str]]
        if all(isinstance(e2, str) for en in entanglement for e2 in en):
            return get_entangler_map(
                n, num_circuit_qubits, entanglement[i % num_i][j % num_j], offset=i
            )

        # entanglement is List[List[int]]
        if all(isinstance(e2, (int, numpy.int32, numpy.int64)) for en in entanglement for e2 in en):
            for ind, en in enumerate(entanglement):
                entanglement[ind] = tuple(map(int, en))
            return entanglement

        # check if entanglement is List[List[List]]
        if not all(isinstance(e2, (tuple, list)) for en in entanglement for e2 in en):
            raise ValueError(f"Invalid value of entanglement: {entanglement}")

        # entanglement is List[List[List[int]]]
        if all(
            isinstance(e3, (int, numpy.int32, numpy.int64))
            for en in entanglement
            for e2 in en
            for e3 in e2
        ):
            for en in entanglement:
                for ind, e2 in enumerate(en):
                    en[ind] = tuple(map(int, e2))
            return entanglement[i % num_i]

        # check if entanglement is List[List[List[List]]]
        if not all(isinstance(e3, (tuple, list)) for en in entanglement for e2 in en for e3 in e2):
            raise ValueError(f"Invalid value of entanglement: {entanglement}")

        # entanglement is List[List[List[List[int]]]]
        if all(
            isinstance(e4, (int, numpy.int32, numpy.int64))
            for en in entanglement
            for e2 in en
            for e3 in e2
            for e4 in e3
        ):
            for en in entanglement:
                for e2 in en:
                    for ind, e3 in enumerate(e2):
                        e2[ind] = tuple(map(int, e3))
            return entanglement[i % num_i][j % num_j]

        raise ValueError(f"Invalid value of entanglement: {entanglement}")

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
        theta = Parameter("θ")
        valid_layers = {
            "ch": CHGate(),
            "cx": CXGate(),
            "cy": CYGate(),
            "cz": CZGate(),
            "crx": CRXGate(theta),
            "cry": CRYGate(theta),
            "crz": CRZGate(theta),
            "h": HGate(),
            "i": IGate(),
            "id": IGate(),
            "iden": IGate(),
            "rx": RXGate(theta),
            "rxx": RXXGate(theta),
            "ry": RYGate(theta),
            "ryy": RYYGate(theta),
            "rz": RZGate(theta),
            "rzx": RZXGate(theta),
            "rzz": RZZGate(theta),
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
