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

"""The n-local circuit class."""

from typing import Union, Optional, List, Any, Tuple, Sequence, Set, Callable
from itertools import combinations
import warnings

import numpy
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit import Instruction, Parameter, ParameterVector, ParameterExpression
from qiskit.circuit.parametertable import ParameterTable
from qiskit.exceptions import QiskitError

from ..blueprintcircuit import BlueprintCircuit


class NLocal(BlueprintCircuit):
    """The n-local circuit class.

    The structure of the n-local circuit are alternating rotation and entanglement layers.
    In both layers, parameterized circuit-blocks act on the circuit in a defined way.
    In the rotation layer, the blocks are applied stacked on top of each other, while in the
    entanglement layer according to the ``entanglement`` strategy.
    The circuit blocks can have arbitrary sizes (smaller equal to the number of qubits in the
    circuit). Each layer is repeated ``reps`` times, and by default a final rotation layer is
    appended.

    For instance, a rotation block on 2 qubits and an entanglement block on 4 qubits using
    ``'linear'`` entanglement yields the following circuit.

    .. parsed-literal::

        ┌──────┐ ░ ┌──────┐                      ░ ┌──────┐
        ┤0     ├─░─┤0     ├──────────────── ... ─░─┤0     ├
        │  Rot │ ░ │      │┌──────┐              ░ │  Rot │
        ┤1     ├─░─┤1     ├┤0     ├──────── ... ─░─┤1     ├
        ├──────┤ ░ │  Ent ││      │┌──────┐      ░ ├──────┤
        ┤0     ├─░─┤2     ├┤1     ├┤0     ├ ... ─░─┤0     ├
        │  Rot │ ░ │      ││  Ent ││      │      ░ │  Rot │
        ┤1     ├─░─┤3     ├┤2     ├┤1     ├ ... ─░─┤1     ├
        ├──────┤ ░ └──────┘│      ││  Ent │      ░ ├──────┤
        ┤0     ├─░─────────┤3     ├┤2     ├ ... ─░─┤0     ├
        │  Rot │ ░         └──────┘│      │      ░ │  Rot │
        ┤1     ├─░─────────────────┤3     ├ ... ─░─┤1     ├
        └──────┘ ░                 └──────┘      ░ └──────┘

        |                                 |
        +---------------------------------+
               repeated reps times

    If specified, barriers can be inserted in between every block.
    If an initial state object of Qiskit Aqua is provided, it is added in front of the NLocal.
    """

    def __init__(
        self,
        num_qubits: Optional[int] = None,
        rotation_blocks: Optional[
            Union[QuantumCircuit, List[QuantumCircuit], Instruction, List[Instruction]]
        ] = None,
        entanglement_blocks: Optional[
            Union[QuantumCircuit, List[QuantumCircuit], Instruction, List[Instruction]]
        ] = None,
        entanglement: Optional[Union[List[int], List[List[int]]]] = None,
        reps: int = 1,
        insert_barriers: bool = False,
        parameter_prefix: str = "θ",
        overwrite_block_parameters: Union[bool, List[List[Parameter]]] = True,
        skip_final_rotation_layer: bool = False,
        skip_unentangled_qubits: bool = False,
        initial_state: Optional[Any] = None,
        name: Optional[str] = "nlocal",
    ) -> None:
        """Create a new n-local circuit.

        Args:
            num_qubits: The number of qubits of the circuit.
            rotation_blocks: The blocks used in the rotation layers. If multiple are passed,
                these will be applied one after another (like new sub-layers).
            entanglement_blocks: The blocks used in the entanglement layers. If multiple are passed,
                these will be applied one after another. To use different entanglements for
                the sub-layers, see :meth:`get_entangler_map`.
            entanglement: The indices specifying on which qubits the input blocks act. If None, the
                entanglement blocks are applied at the top of the circuit.
            reps: Specifies how often the rotation blocks and entanglement blocks are repeated.
            insert_barriers: If True, barriers are inserted in between each layer. If False,
                no barriers are inserted.
            parameter_prefix: The prefix used if default parameters are generated.
            overwrite_block_parameters: If the parameters in the added blocks should be overwritten.
                If False, the parameters in the blocks are not changed.
            skip_final_rotation_layer: Whether a final rotation layer is added to the circuit.
            skip_unentangled_qubits: If ``True``, the rotation gates act only on qubits that
                are entangled. If ``False``, the rotation gates act on all qubits.
            initial_state: A `QuantumCircuit` object which can be used to describe an initial state
                prepended to the NLocal circuit.
            name: The name of the circuit.

        Examples:
            TODO

        Raises:
            ImportError: If an ``initial_state`` is specified but Qiskit Aqua is not installed.
            ValueError: If reps parameter is less than or equal to 0.
            TypeError: If reps parameter is not an int value.
        """
        super().__init__(name=name)

        self._num_qubits = None
        self._insert_barriers = insert_barriers
        self._reps = reps
        self._entanglement_blocks = []
        self._rotation_blocks = []
        self._prepended_blocks = []
        self._prepended_entanglement = []
        self._appended_blocks = []
        self._appended_entanglement = []
        self._entanglement = None
        self._entangler_maps = None
        self._ordered_parameters = ParameterVector(name=parameter_prefix)
        self._overwrite_block_parameters = overwrite_block_parameters
        self._skip_final_rotation_layer = skip_final_rotation_layer
        self._skip_unentangled_qubits = skip_unentangled_qubits
        self._initial_state, self._initial_state_circuit = None, None
        self._data = None
        self._bounds = None

        if int(reps) != reps:
            raise TypeError("The value of reps should be int")

        if reps < 0:
            raise ValueError("The value of reps should be larger than or equal to 0")

        if num_qubits is not None:
            self.num_qubits = num_qubits

        if entanglement_blocks is not None:
            self.entanglement_blocks = entanglement_blocks

        if rotation_blocks is not None:
            self.rotation_blocks = rotation_blocks

        if entanglement is not None:
            self.entanglement = entanglement

        if initial_state is not None:
            self.initial_state = initial_state

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits in this circuit.

        Returns:
            The number of qubits.
        """
        return self._num_qubits if self._num_qubits is not None else 0

    @num_qubits.setter
    def num_qubits(self, num_qubits: int) -> None:
        """Set the number of qubits for the n-local circuit.

        Args:
            The new number of qubits.
        """
        if self._num_qubits != num_qubits:
            # invalidate the circuit
            self._invalidate()
            self._num_qubits = num_qubits
            self.qregs = [QuantumRegister(num_qubits, name="q")]

    def _convert_to_block(self, layer: Any) -> QuantumCircuit:
        """Try to convert ``layer`` to a QuantumCircuit.

        Args:
            layer: The object to be converted to an NLocal block / Instruction.

        Returns:
            The layer converted to a circuit.

        Raises:
            TypeError: If the input cannot be converted to a circuit.
        """
        if isinstance(layer, QuantumCircuit):
            return layer

        if isinstance(layer, Instruction):
            circuit = QuantumCircuit(layer.num_qubits)
            circuit.append(layer, list(range(layer.num_qubits)))
            return circuit

        try:
            circuit = QuantumCircuit(layer.num_qubits)
            circuit.append(layer.to_instruction(), list(range(layer.num_qubits)))
            return circuit
        except AttributeError:
            pass

        raise TypeError(f"Adding a {type(layer)} to an NLocal is not supported.")

    @property
    def rotation_blocks(self) -> List[Instruction]:
        """The blocks in the rotation layers.

        Returns:
            The blocks in the rotation layers.
        """
        return self._rotation_blocks

    @rotation_blocks.setter
    def rotation_blocks(
        self, blocks: Union[QuantumCircuit, List[QuantumCircuit], Instruction, List[Instruction]]
    ) -> None:
        """Set the blocks in the rotation layers.

        Args:
            blocks: The new blocks for the rotation layers.
        """
        # cannot check for the attribute ``'__len__'`` because a circuit also has this attribute
        if not isinstance(blocks, (list, numpy.ndarray)):
            blocks = [blocks]

        self._invalidate()
        self._rotation_blocks = [self._convert_to_block(block) for block in blocks]

    @property
    def entanglement_blocks(self) -> List[Instruction]:
        """The blocks in the entanglement layers.

        Returns:
            The blocks in the entanglement layers.
        """
        return self._entanglement_blocks

    @entanglement_blocks.setter
    def entanglement_blocks(
        self, blocks: Union[QuantumCircuit, List[QuantumCircuit], Instruction, List[Instruction]]
    ) -> None:
        """Set the blocks in the entanglement layers.

        Args:
            blocks: The new blocks for the entanglement layers.
        """
        # cannot check for the attribute ``'__len__'`` because a circuit also has this attribute
        if not isinstance(blocks, (list, numpy.ndarray)):
            blocks = [blocks]

        self._invalidate()
        self._entanglement_blocks = [self._convert_to_block(block) for block in blocks]

    @property
    def entanglement(
        self,
    ) -> Union[
        str,
        List[str],
        List[List[str]],
        List[int],
        List[List[int]],
        List[List[List[int]]],
        List[List[List[List[int]]]],
        Callable[[int], str],
        Callable[[int], List[List[int]]],
    ]:
        """Get the entanglement strategy.

        Returns:
            The entanglement strategy, see :meth:`get_entangler_map` for more detail on how the
            format is interpreted.
        """
        return self._entanglement

    @entanglement.setter
    def entanglement(
        self,
        entanglement: Optional[
            Union[
                str,
                List[str],
                List[List[str]],
                List[int],
                List[List[int]],
                List[List[List[int]]],
                List[List[List[List[int]]]],
                Callable[[int], str],
                Callable[[int], List[List[int]]],
            ]
        ],
    ) -> None:
        """Set the entanglement strategy.

        Args:
            entanglement: The entanglement strategy. See :meth:`get_entangler_map` for more detail
                on the supported formats.
        """
        self._invalidate()
        self._entanglement = entanglement

    @property
    def num_layers(self) -> int:
        """Return the number of layers in the n-local circuit.

        Returns:
            The number of layers in the circuit.
        """
        return 2 * self._reps + int(not self._skip_final_rotation_layer)

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Check if the configuration of the NLocal class is valid.

        Args:
            raise_on_failure: Whether to raise on failure.

        Returns:
            True, if the configuration is valid and the circuit can be constructed. Otherwise
            an ValueError is raised.

        Raises:
            ValueError: If the blocks are not set.
            ValueError: If the number of repetitions is not set.
            ValueError: If the qubit indices are not set.
            ValueError: If the number of qubit indices does not match the number of blocks.
            ValueError: If an index in the repetitions list exceeds the number of blocks.
            ValueError: If the number of repetitions does not match the number of block-wise
                parameters.
            ValueError: If a specified qubit index is larger than the (manually set) number of
                qubits.
        """
        valid = True
        if self.num_qubits is None:
            valid = False
            if raise_on_failure:
                raise ValueError("No number of qubits specified.")

        # check no needed parameters are None
        if self.entanglement_blocks is None and self.rotation_blocks is None:
            valid = False
            if raise_on_failure:
                raise ValueError("The blocks are not set.")

        return valid

    @property
    def ordered_parameters(self) -> List[Parameter]:
        """The parameters used in the underlying circuit.

        This includes float values and duplicates.

        Examples:

            >>> # prepare circuit ...
            >>> print(nlocal)
                 ┌───────┐┌──────────┐┌──────────┐┌──────────┐
            q_0: ┤ Ry(1) ├┤ Ry(θ[1]) ├┤ Ry(θ[1]) ├┤ Ry(θ[3]) ├
                 └───────┘└──────────┘└──────────┘└──────────┘
            >>> nlocal.parameters
            {Parameter(θ[1]), Parameter(θ[3])}
            >>> nlocal.ordered_parameters
            [1, Parameter(θ[1]), Parameter(θ[1]), Parameter(θ[3])]

        Returns:
            The parameters objects used in the circuit.
        """
        if isinstance(self._ordered_parameters, ParameterVector):
            self._ordered_parameters.resize(self.num_parameters_settable)
            return list(self._ordered_parameters)

        return self._ordered_parameters

    @ordered_parameters.setter
    def ordered_parameters(self, parameters: Union[ParameterVector, List[Parameter]]) -> None:
        """Set the parameters used in the underlying circuit.

        Args:
            The parameters to be used in the underlying circuit.

        Raises:
            ValueError: If the length of ordered parameters does not match the number of
                parameters in the circuit and they are not a ``ParameterVector`` (which could
                be resized to fit the number of parameters).
        """
        if (
            not isinstance(parameters, ParameterVector)
            and len(parameters) != self.num_parameters_settable
        ):
            raise ValueError(
                "The length of ordered parameters must be equal to the number of "
                "settable parameters in the circuit ({}), but is {}".format(
                    self.num_parameters_settable, len(parameters)
                )
            )
        self._ordered_parameters = parameters
        self._invalidate()

    @property
    def insert_barriers(self) -> bool:
        """If barriers are inserted in between the layers or not.

        Returns:
            True, if barriers are inserted in between the layers, False if not.
        """
        return self._insert_barriers

    @insert_barriers.setter
    def insert_barriers(self, insert_barriers: bool) -> None:
        """Specify whether barriers should be inserted in between the layers or not.

        Args:
            insert_barriers: If True, barriers are inserted, if False not.
        """
        # if insert_barriers changes, we have to invalidate the circuit definition,
        # if it is the same as before we can leave the NLocal instance as it is
        if insert_barriers is not self._insert_barriers:
            self._invalidate()
            self._insert_barriers = insert_barriers

    def get_unentangled_qubits(self) -> Set[int]:
        """Get the indices of unentangled qubits in a set.

        Returns:
            The unentangled qubits.
        """
        entangled_qubits = set()
        for i in range(self._reps):
            for j, block in enumerate(self.entanglement_blocks):
                entangler_map = self.get_entangler_map(i, j, block.num_qubits)
                entangled_qubits.update([idx for indices in entangler_map for idx in indices])
        unentangled_qubits = set(range(self.num_qubits)) - entangled_qubits

        return unentangled_qubits

    @property
    def num_parameters_settable(self) -> int:
        """The number of total parameters that can be set to distinct values.

        This does not change when the parameters are bound or exchanged for same parameters,
        and therefore is different from ``num_parameters`` which counts the number of unique
        :class:`~qiskit.circuit.Parameter` objects currently in the circuit.

        Returns:
            The number of parameters originally available in the circuit.

        Note:
            This quantity does not require the circuit to be built yet.
        """
        num = 0

        for i in range(self._reps):
            for j, block in enumerate(self.entanglement_blocks):
                entangler_map = self.get_entangler_map(i, j, block.num_qubits)
                num += len(entangler_map) * len(get_parameters(block))

        if self._skip_unentangled_qubits:
            unentangled_qubits = self.get_unentangled_qubits()

        num_rot = 0
        for block in self.rotation_blocks:
            block_indices = [
                list(range(j * block.num_qubits, (j + 1) * block.num_qubits))
                for j in range(self.num_qubits // block.num_qubits)
            ]
            if self._skip_unentangled_qubits:
                block_indices = [
                    indices
                    for indices in block_indices
                    if set(indices).isdisjoint(unentangled_qubits)
                ]
            num_rot += len(block_indices) * len(get_parameters(block))

        num += num_rot * (self._reps + int(not self._skip_final_rotation_layer))

        return num

    @property
    def reps(self) -> int:
        """The number of times rotation and entanglement block are repeated.

        Returns:
            The number of repetitions.
        """
        return self._reps

    @reps.setter
    def reps(self, repetitions: int) -> None:
        """Set the repetitions.

        If the repetitions are `0`, only one rotation layer with no entanglement
        layers is applied (unless ``self.skip_final_rotation_layer`` is set to ``True``).

        Args:
            repetitions: The new repetitions.

        Raises:
            ValueError: If reps setter has parameter repetitions < 0.
        """
        if repetitions < 0:
            raise ValueError("The repetitions should be larger than or equal to 0")
        if repetitions != self._reps:
            self._invalidate()
            self._reps = repetitions

    def print_settings(self) -> str:
        """Returns information about the setting.

        Returns:
            The class name and the attributes/parameters of the instance as ``str``.
        """
        ret = f"NLocal: {self.__class__.__name__}\n"
        params = ""
        for key, value in self.__dict__.items():
            if key[0] == "_":
                params += f"-- {key[1:]}: {value}\n"
        ret += f"{params}"
        return ret

    @property
    def preferred_init_points(self) -> Optional[List[float]]:
        """The initial points for the parameters. Can be stored as initial guess in optimization.

        Returns:
            The initial values for the parameters, or None, if none have been set.
        """
        return None

    # pylint: disable=too-many-return-statements
    def get_entangler_map(
        self, rep_num: int, block_num: int, num_block_qubits: int
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

        Returns:
            The entangler map for the current block in the current repetition.

        Raises:
            ValueError: If the value of ``entanglement`` could not be cast to a corresponding
                entangler map.
        """
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
            return get_entangler_map(n, self.num_qubits, entanglement, offset=i)

        # check if entanglement is list of something
        if not isinstance(entanglement, (tuple, list)):
            raise ValueError(f"Invalid value of entanglement: {entanglement}")
        num_i = len(entanglement)

        # entanglement is List[str]
        if all(isinstance(en, str) for en in entanglement):
            return get_entangler_map(n, self.num_qubits, entanglement[i % num_i], offset=i)

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
                n, self.num_qubits, entanglement[i % num_i][j % num_j], offset=i
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

    @property
    def initial_state(self) -> Any:
        """Return the initial state that is added in front of the n-local circuit.

        Returns:
            The initial state.
        """
        return self._initial_state

    @initial_state.setter
    def initial_state(self, initial_state: Any) -> None:
        """Set the initial state.

        Args:
            initial_state: The new initial state.

        Raises:
            ValueError: If the number of qubits has been set before and the initial state
                does not match the number of qubits.
        """
        self._initial_state = initial_state

        # If there is an initial state object, check that the number of qubits is compatible
        # construct the circuit immediately. If the InitialState could modify the number of qubits
        # we could also do this later at circuit construction.
        if not isinstance(self._initial_state, QuantumCircuit):
            warnings.warn(
                "The initial_state argument of the NLocal class "
                "should be a QuantumCircuit. Passing any other type is "
                "deprecated as of Qiskit Terra 0.18.0, and "
                "will be removed no earlier than 3 months after that "
                "release date.",
                DeprecationWarning,
                stacklevel=2,
            )
            initial_state_circuit = initial_state.construct_circuit(mode="circuit")

            # the initial state dictates the number of qubits since we do not have information
            # about on which qubits the initial state acts
            if self._num_qubits is not None and initial_state_circuit.num_qubits != self.num_qubits:
                raise ValueError(
                    "Mismatching number of qubits in initial state and n-local circuit."
                )

        self._invalidate()

    @property
    def parameter_bounds(self) -> Optional[List[Tuple[float, float]]]:
        """The parameter bounds for the unbound parameters in the circuit.

        Returns:
            A list of pairs indicating the bounds, as (lower, upper). None indicates an unbounded
            parameter in the corresponding direction. If None is returned, problem is fully
            unbounded.
        """
        self._build()
        return self._bounds

    @parameter_bounds.setter
    def parameter_bounds(self, bounds: List[Tuple[float, float]]) -> None:
        """Set the parameter bounds.

        Args:
            bounds: The new parameter bounds.
        """
        self._bounds = bounds

    def _invalidate(self):
        """Invalidate the current circuit build."""
        self._data = None
        self._parameter_table = ParameterTable()

    def add_layer(
        self,
        other: Union["NLocal", Instruction, QuantumCircuit],
        entanglement: Optional[Union[List[int], str, List[List[int]]]] = None,
        front: bool = False,
    ) -> "NLocal":
        """Append another layer to the NLocal.

        Args:
            other: The layer to compose, can be another NLocal, an Instruction or Gate,
                or a QuantumCircuit.
            entanglement: The entanglement or qubit indices.
            front: If True, ``other`` is appended to the front, else to the back.

        Returns:
            self, such that chained composes are possible.

        Raises:
            TypeError: If `other` is not compatible, i.e. is no Instruction and does not have a
                `to_instruction` method.
        """
        block = self._convert_to_block(other)

        if entanglement is None:
            entanglement = [list(range(block.num_qubits))]
        elif isinstance(entanglement, list) and not isinstance(entanglement[0], list):
            entanglement = [entanglement]

        if front:
            self._prepended_blocks += [block]
            self._prepended_entanglement += [entanglement]
        else:
            self._appended_blocks += [block]
            self._appended_entanglement += [entanglement]

        if isinstance(entanglement, list):
            num_qubits = 1 + max(max(indices) for indices in entanglement)
            if num_qubits > self.num_qubits:
                self._invalidate()  # rebuild circuit
                self.num_qubits = num_qubits

        # modify the circuit accordingly
        if self._data and front is False:
            if self._insert_barriers and len(self._data) > 0:
                self.barrier()

            if isinstance(entanglement, str):
                entangler_map = get_entangler_map(block.num_qubits, self.num_qubits, entanglement)
            else:
                entangler_map = entanglement

            layer = QuantumCircuit(self.num_qubits)
            for i in entangler_map:
                params = self.ordered_parameters[-len(get_parameters(block)) :]
                parameterized_block = self._parameterize_block(block, params=params)
                layer.compose(parameterized_block, i)

            self.compose(layer, inplace=True)
        else:
            # cannot prepend a block currently, just rebuild
            self._invalidate()

        return self

    def assign_parameters(
        self,
        parameters: Union[dict, List[float], List[Parameter], ParameterVector],
        inplace: bool = False,
    ) -> Optional[QuantumCircuit]:
        """Assign parameters to the n-local circuit.

        This method also supports passing a list instead of a dictionary. If a list
        is passed, the list must have the same length as the number of unbound parameters in
        the circuit. The parameters are assigned in the order of the parameters in
        :meth:`ordered_parameters`.

        Returns:
            A copy of the NLocal circuit with the specified parameters.

        Raises:
            AttributeError: If the parameters are given as list and do not match the number
                of parameters.
        """
        if self._data is None:
            self._build()

        return super().assign_parameters(parameters, inplace=inplace)

    def _parameterize_block(
        self, block, param_iter=None, rep_num=None, block_num=None, indices=None, params=None
    ):
        """Convert ``block`` to a circuit of correct width and parameterized using the iterator."""
        if self._overwrite_block_parameters:
            # check if special parameters should be used
            # pylint: disable=assignment-from-none
            if params is None:
                params = self._parameter_generator(rep_num, block_num, indices)
            if params is None:
                params = [next(param_iter) for _ in range(len(get_parameters(block)))]

            update = dict(zip(block.parameters, params))
            return block.assign_parameters(update)

        return block.copy()

    def _build_rotation_layer(self, circuit, param_iter, i):
        """Build a rotation layer."""
        # if the unentangled qubits are skipped, compute the set of qubits that are not entangled
        if self._skip_unentangled_qubits:
            unentangled_qubits = self.get_unentangled_qubits()

        # iterate over all rotation blocks
        for j, block in enumerate(self.rotation_blocks):
            # create a new layer
            layer = QuantumCircuit(*self.qregs)

            # we apply the rotation gates stacked on top of each other, i.e.
            # if we have 4 qubits and a rotation block of width 2, we apply two instances
            block_indices = [
                list(range(k * block.num_qubits, (k + 1) * block.num_qubits))
                for k in range(self.num_qubits // block.num_qubits)
            ]

            # if unentangled qubits should not be acted on, remove all operations that
            # touch an unentangled qubit
            if self._skip_unentangled_qubits:
                block_indices = [
                    indices
                    for indices in block_indices
                    if set(indices).isdisjoint(unentangled_qubits)
                ]

            # apply the operations in the layer
            for indices in block_indices:
                parameterized_block = self._parameterize_block(block, param_iter, i, j, indices)
                layer.compose(parameterized_block, indices, inplace=True)

            # add the layer to the circuit
            circuit.compose(layer, inplace=True)

    def _build_entanglement_layer(self, circuit, param_iter, i):
        """Build an entanglement layer."""
        # iterate over all entanglement blocks
        for j, block in enumerate(self.entanglement_blocks):
            # create a new layer and get the entangler map for this block
            layer = QuantumCircuit(*self.qregs)
            entangler_map = self.get_entangler_map(i, j, block.num_qubits)

            # apply the operations in the layer
            for indices in entangler_map:
                parameterized_block = self._parameterize_block(block, param_iter, i, j, indices)
                layer.compose(parameterized_block, indices, inplace=True)

            # add the layer to the circuit
            circuit.compose(layer, inplace=True)

    def _build_additional_layers(self, circuit, which):
        if which == "appended":
            blocks = self._appended_blocks
            entanglements = self._appended_entanglement
        elif which == "prepended":
            blocks = reversed(self._prepended_blocks)
            entanglements = reversed(self._prepended_entanglement)
        else:
            raise ValueError("`which` must be either `appended` or `prepended`.")

        for block, ent in zip(blocks, entanglements):
            layer = QuantumCircuit(*self.qregs)
            if isinstance(ent, str):
                ent = get_entangler_map(block.num_qubits, self.num_qubits, ent)
            for indices in ent:
                layer.compose(block, indices, inplace=True)

            circuit.compose(layer, inplace=True)

    def _build(self) -> None:
        """Build the circuit."""
        if self._data:
            return

        _ = self._check_configuration()

        self._data = []

        if self.num_qubits == 0:
            return

        circuit = QuantumCircuit(*self.qregs, name=self.name)

        # use the initial state as starting circuit, if it is set
        if self.initial_state:
            if isinstance(self.initial_state, QuantumCircuit):
                initial = self.initial_state.copy()
            else:
                initial = self.initial_state.construct_circuit("circuit", register=self.qregs[0])
            circuit.compose(initial, inplace=True)

        param_iter = iter(self.ordered_parameters)

        # build the prepended layers
        self._build_additional_layers(circuit, "prepended")

        # main loop to build the entanglement and rotation layers
        for i in range(self.reps):
            # insert barrier if specified and there is a preceding layer
            if self._insert_barriers and (i > 0 or len(self._prepended_blocks) > 0):
                circuit.barrier()

            # build the rotation layer
            self._build_rotation_layer(circuit, param_iter, i)

            # barrier in between rotation and entanglement layer
            if self._insert_barriers and len(self._rotation_blocks) > 0:
                circuit.barrier()

            # build the entanglement layer
            self._build_entanglement_layer(circuit, param_iter, i)

        # add the final rotation layer
        if not self._skip_final_rotation_layer:
            if self.insert_barriers and self.reps > 0:
                circuit.barrier()
            self._build_rotation_layer(circuit, param_iter, self.reps)

        # add the appended layers
        self._build_additional_layers(circuit, "appended")

        # cast global phase to float if it has no free parameters
        if isinstance(circuit.global_phase, ParameterExpression):
            try:
                circuit.global_phase = float(circuit.global_phase._symbol_expr)
            # RuntimeError is raised if symengine is used, for SymPy it is a TypeError
            except (RuntimeError, TypeError):
                # expression contains free parameters
                pass

        try:
            block = circuit.to_gate()
        except QiskitError:
            block = circuit.to_instruction()

        self.append(block, self.qubits)

    # pylint: disable=unused-argument
    def _parameter_generator(self, rep: int, block: int, indices: List[int]) -> Optional[Parameter]:
        """If certain blocks should use certain parameters this method can be overriden."""
        return None

    def __str__(self) -> str:
        """Draw this NLocal in circuit format using the standard gates.

        Returns:
            A single string representing this NLocal.
        """
        from qiskit.compiler import transpile

        basis_gates = [
            "id",
            "x",
            "y",
            "z",
            "h",
            "s",
            "t",
            "sdg",
            "tdg",
            "rx",
            "ry",
            "rz",
            "rxx",
            "ryy",
            "cx",
            "cy",
            "cz",
            "ch",
            "crx",
            "cry",
            "crz",
            "swap",
            "cswap",
            "ccx",
            "cu1",
            "cu3",
            "u1",
            "u2",
            "u3",
        ]
        return (
            transpile(self, basis_gates=basis_gates, optimization_level=0)
            .draw(output="text")
            .single_string()
        )


def get_parameters(block: Union[QuantumCircuit, Instruction]) -> List[Parameter]:
    """Return the list of Parameters objects inside a circuit or instruction.

    This is required since, in a standard gate the parameters are not necessarily Parameter
    objects (e.g. U3Gate(0.1, 0.2, 0.3).params == [0.1, 0.2, 0.3]) and instructions and
    circuits do not have the same interface for parameters.
    """
    if isinstance(block, QuantumCircuit):
        return list(block.parameters)
    else:
        return [p for p in block.params if isinstance(p, ParameterExpression)]


def get_entangler_map(
    num_block_qubits: int, num_circuit_qubits: int, entanglement: str, offset: int = 0
) -> List[Sequence[int]]:
    """Get an entangler map for an arbitrary number of qubits.

    Args:
        num_block_qubits: The number of qubits of the entangling block.
        num_circuit_qubits: The number of qubits of the circuit.
        entanglement: The entanglement strategy.
        offset: The block offset, can be used if the entanglements differ per block.
            See mode ``sca`` for instance.

    Returns:
        The entangler map using mode ``entanglement`` to scatter a block of ``num_block_qubits``
        qubits on ``num_circuit_qubits`` qubits.

    Raises:
        ValueError: If the entanglement mode ist not supported.
    """
    n, m = num_circuit_qubits, num_block_qubits
    if m > n:
        raise ValueError(
            "The number of block qubits must be smaller or equal to the number of "
            "qubits in the circuit."
        )

    if entanglement == "pairwise" and num_block_qubits != 2:
        raise ValueError("Pairwise entanglement is only defined for blocks of 2 qubits.")

    if entanglement == "full":
        return list(combinations(list(range(n)), m))
    if entanglement in ["linear", "circular", "sca", "pairwise"]:
        linear = [tuple(range(i, i + m)) for i in range(n - m + 1)]
        # if the number of block qubits is 1, we don't have to add the 'circular' part
        if entanglement == "linear" or m == 1:
            return linear

        if entanglement == "pairwise":
            return linear[::2] + linear[1::2]

        # circular equals linear plus top-bottom entanglement (if there's space for it)
        if n > m:
            circular = [tuple(range(n - m + 1, n)) + (0,)] + linear
        else:
            circular = linear
        if entanglement == "circular":
            return circular

        # sca is circular plus shift and reverse
        shifted = circular[-offset:] + circular[:-offset]
        if offset % 2 == 1:  # if odd, reverse the qubit indices
            sca = [ind[::-1] for ind in shifted]
        else:
            sca = shifted

        return sca

    else:
        raise ValueError(f"Unsupported entanglement type: {entanglement}")
