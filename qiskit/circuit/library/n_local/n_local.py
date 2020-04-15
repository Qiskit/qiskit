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

"""The NLocal class.

TODO
    * store circuits instead of gates?
        - Reverting to circuits in future anyways
        - Performance difference?
    * add transpile feature
    * rename append to combine(after=True) with to support after/before
"""

# import copy
import warnings
import logging
from typing import Union, Optional, List, Any, Tuple, Sequence, Set
from itertools import combinations

import numpy
from qiskit import QuantumCircuit, transpile, QuantumRegister
from qiskit.circuit import Instruction, Parameter, ParameterVector, ParameterExpression
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.parametertable import ParameterTable

logger = logging.getLogger(__name__)


class NLocal(QuantumCircuit):
    """The n-local circuit class."""

    def __init__(self,
                 num_qubits: Optional[int] = None,
                 rotation_blocks: Optional[Union[QuantumCircuit, List[QuantumCircuit],
                                                 Instruction, List[Instruction]]] = None,
                 entanglement_blocks: Optional[Union[QuantumCircuit, List[QuantumCircuit],
                                                     Instruction, List[Instruction]]] = None,
                 entanglement: Optional[Union[List[int], List[List[int]]]] = None,
                 reps: int = 1,
                 insert_barriers: bool = False,
                 parameter_prefix: str = 'θ',
                 overwrite_block_parameters: Union[bool, List[List[Parameter]]] = True,
                 skip_final_rotation_layer: bool = False,
                 initial_state: Optional['InitialState'] = None) -> None:
        """Create a new n-local circuit.

        The structure of the NLocal are repeated parameterized circuit-blocks (referred to
        as blocks in the code). For every block, qubit indices indicate on which qubits the block
        acts on. Which and how blocks are repeated is determined via the ``reps`` argument.
        If the same block is supposed to be repeated several times with the same qubit indices,
        it is only stored once.  On circuit construction, copies of the blocks are inserted in the
        NLocal with new, unique parameters.
        These "base parameters" do not change. The user modifies "surface parameters" which are
        bound to the circuit upon requesting the circuit.
        If specified, barriers can be inserted in between every block.
        If an initial state is provided, it is added in front of the NLocal.

        Args:
            num_qubits: The number of qubits of the circuit.
            rotation_blocks: The rotation blocks.
            entanglement_blocks: The entanglement blocks.
            entanglement: The indices specifying on which qubits the input blocks act. If None, for
                each block this is set to the first ``n`` qubits, where ``n`` is the number of
                qubits the block acts on.
            reps: Specifies how the input blocks are repeated. If an integer, all input blocks
                are repeated ``reps`` times (in the provided order). If a list of
                integers, ``reps`` determines the order of the layers in NLocal using the
                elements of ``reps`` as index. See the Examples section for more detail.
            insert_barriers: If True, barriers are inserted in between each layer/block. If False,
                no barriers are inserted.
            parameter_prefix: The prefix used if default parameters are generated.
            overwrite_block_parameters: If the parameters in the added blocks should be overwritten.
                If a list of list of Parameters is passed, these Parameters are used to set the
                parameters in the blocks.
            skip_final_rotation_layer: Whether a final rotation layer is added to the circuit.
            initial_state: An ``InitialState`` object to prepend to the NLocal.
                TODO deprecate this feature in favor of prepend or overloading __add__ in
                the initial state class

        Raises:
            TypeError: If ``blocks`` contains an unsupported object.
            ValueError: If the initial state has less qubits than specified via the blocks or
                qubit indices.
            ValueError: If the ``overwrite_block_parameters`` is set to a list of list of
                Parameters but does not match the total number of blocks in the final circuit.

        Examples:
            TODO
        """
        super().__init__()

        # insert barriers in between the blocks?
        self._num_qubits = None
        self.num_qubits = num_qubits

        self._insert_barriers = insert_barriers

        self._entanglement_blocks = []
        self._rotation_blocks = []
        self._prepended_blocks = []
        self._prepended_entanglement = []
        self._appended_blocks = []
        self._appended_entanglement = []

        self.entanglement_blocks = entanglement_blocks or []
        self.rotation_blocks = rotation_blocks or []
        self._ordered_parameters = ParameterVector(name=parameter_prefix)
        self._overwrite_block_parameters = overwrite_block_parameters
        self._skip_final_rotation_layer = skip_final_rotation_layer

        # get reps in the right format
        self._reps = reps

        # get entanglement in the right format (i.e. list of lists)
        self._entanglement, self._entangler_maps = None, None
        self.entanglement = entanglement
        # self._entangler_maps = None

        # set the initial state
        self._initial_state, self._initial_state_circuit = None, None
        if initial_state:
            self.initial_state = initial_state

        # keep track of the circuit
        self._data = None

        # parameter bounds
        self._bounds = None

    def __str__(self) -> str:
        """Draw this NLocal in circuit format using the standard gates.

        Returns:
            A single string representing this NLocal.
        """
        basis_gates = ['id', 'x', 'y', 'z', 'h', 's', 't', 'sdg', 'tdg', 'rx', 'ry', 'rz',
                       'rxx', 'ryy', 'cx', 'cy', 'cz', 'ch', 'crx', 'cry', 'crz', 'swap',
                       'cswap', 'ccx', 'cu1', 'cu3', 'u1', 'u2', 'u3']
        return transpile(self.to_circuit(), basis_gates=basis_gates,
                         optimization_level=0).draw(fold=1000).single_string()

    def _convert_to_block(self, layer: Any) -> Instruction:
        """Try to convert ``layer`` to an Instruction.

        Args:
            layer: The object to be converted to an NLocal block / Instruction.

        Returns:
            The layer converted to an Instruction.

        Raises:
            TypeError: If the input cannot be converted to an Instruction.
        """
        if isinstance(layer, Instruction):
            return layer
        elif hasattr(layer, 'to_instruction'):
            return layer.to_instruction()
        else:
            raise TypeError('Adding a {} to an NLocal is not supported.'.format(type(layer)))

    def _configuration_is_valid(self) -> bool:
        """Check if the configuration of the NLocal class is valid.

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
        if self.num_qubits is None:
            raise ValueError('No number of qubits specified.')

        # check no needed parameters are None
        if self.entanglement_blocks is None and self.rotation_blocks is None:
            raise ValueError('The blocks are not set.')

        # check the compatibility of the attributes
        if len(self.entanglement_blocks) > 0 and len(self.entangler_maps) != self._reps:
            raise ValueError('The number of qubit indices does not match the number of '
                             'repetitions.')

        if isinstance(self._reps, list):
            if len(self._reps) > 0:
                if max(self._reps) >= len(self._blocks):
                    raise ValueError('Trying to add a non-existing block to the circuit.')

        if self._num_qubits:
            for layer in self.entangler_maps:
                for block in layer:
                    if max(block) >= self._num_qubits:
                        raise ValueError('The manually set number of qubits is too small for the '
                                         + 'blocks in the circuit.')

        return True

    def _rotation_reps(self) -> List[int]:
        """Return the indices of the rotation layers that go in the circuit."""
        if self.rotation_blocks is None or len(self.rotation_blocks) == 0:
            return []
        # + 1 for final rotation layer
        return [i % len(self.rotation_blocks) for i in range(self._reps + 1)]

    def _entanglement_reps(self) -> List[int]:
        """Return the indices of the entanglement layers that go in the circuit."""
        if self.entanglement_blocks is None or len(self.entanglement_blocks) == 0:
            return []
        return [i % len(self.entanglement_blocks) for i in range(self._reps)]

    def print_settings(self) -> str:
        """Returns information about the setting.

        Returns:
            The class name and the attributes/parameters of the instance as ``str``.
        """
        ret = 'NLocal: {}\n'.format(self.__class__.__name__)
        params = ''
        for key, value in self.__dict__.items():
            if key[0] == '_':
                params += '-- {}: {}\n'.format(key[1:], value)
        ret += '{}'.format(params)
        return ret

    @property
    def setting(self):
        """Deprecated, moved to __str__."""
        return self.print_settings()

    @property
    def preferred_init_points(self):
        """TODO Deprecate.

        Returns preferred init points."""
        return None

    @property
    def support_parameterized_circuit(self):
        """TODO Deprecate.

        Whether it is supported to bind parameters in this circuit.
        """
        return True

    @property
    def ordered_parameters(self) -> List[Parameter]:
        """The parameters used in the underlying circuit.

        Returns:
            The parameters objects used in the circuit.

        Raises:
            CircuitError: If the length of ordered parameters does not match the number of
                parameters in the circuit and they are not a ``ParameterVector`` (which could
                be resized to fit the number of parameters).
        """
        if isinstance(self._ordered_parameters, ParameterVector):
            self._ordered_parameters.resize(self.num_parameters_settable)
            return list(self._ordered_parameters)

        if len(self._ordered_parameters) != self.num_parameters_settable:
            raise CircuitError('The number of ordered parameters does not match the number '
                               'of circuit parameters, {} != {}'.format(
                                   len(self._ordered_parameters), self.num_parameters_settable
                               ))
        return self._ordered_parameters

    @ordered_parameters.setter
    def ordered_parameters(self, parameters: Union[ParameterVector, List[Parameter]]) -> None:
        """Set the parameters used in the underlying circuit.

        Args:
            The parameters to be used in the underlying circuit.
        """
        self.assign_parameters(dict(zip(self._ordered_parameters, parameters)))

    @property
    def entanglement_blocks(self) -> List[Instruction]:
        """The blocks in the NLocal.

        Returns:
            The blocks that define the NLocal.
        """
        return self._entanglement_blocks

    @property
    def rotation_blocks(self) -> List[Instruction]:
        """The blocks in the NLocal.

        Returns:
            The blocks that define the NLocal.
        """
        return self._rotation_blocks

    @rotation_blocks.setter
    def rotation_blocks(self, blocks: Union[QuantumCircuit, List[QuantumCircuit],
                                            Instruction, List[Instruction]]) -> None:
        """Set the blocks of the NLocal.

        Args:
            blocks: The new blocks of the NLocal.
        """
        # cannot check for the attribute ``'__len__'`` because a circuit also has this attribute
        if not isinstance(blocks, (list, numpy.ndarray)):
            blocks = [blocks]

        self._invalidate()
        self._rotation_blocks = [self._convert_to_block(block) for block in blocks]

    @entanglement_blocks.setter
    def entanglement_blocks(self, blocks: Union[QuantumCircuit, List[QuantumCircuit],
                                                Instruction, List[Instruction]]) -> None:
        """Set the blocks of the NLocal.

        Args:
            blocks: The new blocks of the NLocal.
        """
        # cannot check for the attribute ``'__len__'`` because a circuit also has this attribute
        if not isinstance(blocks, (list, numpy.ndarray)):
            blocks = [blocks]

        self._invalidate()
        self._entanglement_blocks = [self._convert_to_block(block) for block in blocks]

    # pylint:disable=too-many-return-statements
    def get_entangler_map(self, rep_num: int, block_num: int, num_block_qubits: int
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
        if not isinstance(entanglement, list):
            raise ValueError('Invalid value of entanglement: {}'.format(entanglement))
        num_i = len(entanglement)

        # entanglement is List[str]
        if all(isinstance(e, str) for e in entanglement):
            return get_entangler_map(n, self.num_qubits, entanglement[i % num_i], offset=i)

        # entanglement is List[int]
        if all(isinstance(e, int) for e in entanglement):
            return [entanglement]

        # check if entanglement is List[List]
        if not all(isinstance(e, list) for e in entanglement):
            raise ValueError('Invalid value of entanglement: {}'.format(entanglement))
        num_j = len(entanglement[i])

        # entanglement is List[List[str]]
        if all(isinstance(e2, str) for e in entanglement for e2 in e):
            return get_entangler_map(n, self.num_qubits, entanglement[i % num_i][j % num_j],
                                     offset=i)

        # entanglement is List[List[int]]
        if all(isinstance(e2, str) for e in entanglement for e2 in e):
            return entanglement

        # check if entanglement is List[List[List]]
        if not all(isinstance(e2, list) for e in entanglement for e2 in e):
            raise ValueError('Invalid value of entanglement: {}'.format(entanglement))

        # entanglement is List[List[List[int]]]
        if all(isinstance(e3, int) for e in entanglement for e2 in e for e3 in e2):
            return entanglement[i % num_i]

        # check if entanglement is List[List[List[List]]]
        if not all(isinstance(e3, list) for e in entanglement for e2 in e for e3 in e2):
            raise ValueError('Invalid value of entanglement: {}'.format(entanglement))

        # entanglement is List[List[List[List[int]]]]
        if all(isinstance(e4, int) for e in entanglement for e2 in e for e3 in e2 for e4 in e3):
            return entanglement[i % num_i][j % num_j]

        raise ValueError('Invalid value of entanglement: {}'.format(entanglement))

    @property
    def entangler_maps(self) -> List[List[Sequence[int]]]:
        """The qubit indices each block acts on.

        An entangler map is a list of qubit indicies (tuples of ints) that specify on which qubits
        the block act. If the block acts on two qubits (e.g. a CX gate) then the entangler map
        is a list of 2-tuples, e.g. ``[(0, 1), (0, 2), (1, 2)]``.
        This property returns a list of such entangler maps, one entangler map per block.

        Returns:
            A list of entangler maps for each block.

        Raises:
            RuntimeError: If the format of the entanglement cannot be understood.
        """
        # if no entanglement was set return default
        entanglement = self._entanglement
        if not entanglement:
            return [[list(range(self.entanglement_blocks[i].num_qubits))]
                    for i in self._entanglement_reps()]

        if isinstance(entanglement, str):
            entangler_maps = []
            for num, i in enumerate(self._entanglement_reps()):
                block = self.entanglement_blocks[i]
                entangler_maps += [
                    get_entangler_map(block.num_qubits, self.num_qubits, entanglement, num)
                ]
            return entangler_maps

        if callable(entanglement):
            entangler_maps = []
            for num, i in enumerate(self._entanglement_reps()):
                ent = entanglement(num)
                block = self.entanglement_blocks[i]
                if isinstance(entanglement, str):
                    entangler_maps += [
                        get_entangler_map(block.num_qubits, self.num_qubits, ent, num)
                    ]
                else:
                    entangler_maps += [ent]

        if isinstance(entanglement, list):
            # is list of strings
            if all(isinstance(e, str) for e in entanglement):
                entangler_maps = []
                for num, i, ent in enumerate(zip(self._entanglement_reps(), entanglement)):
                    block = self.entanglement_blocks[i]
                    entangler_maps += [
                        get_entangler_map(block.num_qubits, self.num_qubits, ent, num)
                    ]
                    return entangler_maps

            if all(isinstance(e, (tuple, list)) for e in entanglement):
                # is list of lists of int, i.e. a single entangler map
                if all(isinstance(e_i, int) for e in entanglement for e_i in e):
                    return [entanglement] * self.num_layers

                # is list of lists of lists of int, i.e. a list of entangler maps
                if all(isinstance(e_i, (tuple, list)) for e in entanglement for e_i in e):
                    if all(isinstance(e_j, int) for e in entanglement for e_i in e for e_j in e_i):
                        return entanglement

        raise RuntimeError('Could not understand format of entanglement: {}'.format(entanglement))

    @property
    def entanglement(self):
        """Get the entanglement strategy."""
        return self._entanglement

    @entanglement.setter
    def entanglement(self, entanglement: Optional[Union[str, List[str], List[List[int]],
                                                        List[List[List[int]]]]]) -> None:
        """Set the entanglement strategy."""
        valid_format = False
        # TODO set entangler maps correctly
        if entanglement is None:
            valid_format = True
        elif callable(entanglement):
            # no real checking possible here, but that happens later when the entangler maps
            # are generated and the configuration to construct the circuit is checked
            valid_format = True
        elif isinstance(entanglement, str):
            valid_format = True
        elif isinstance(entanglement, list):
            if all(isinstance(e, str) for e in entanglement):
                valid_format = True
            elif all(isinstance(e, (tuple, list)) for e in entanglement):  # is List[List[?]]
                if all(isinstance(e_i, int) for e in entanglement for e_i in e):
                    valid_format = True
                elif all(isinstance(e_i, (tuple, list)) for e in entanglement for e_i in e):
                    if all(isinstance(e_j, int) for e in entanglement for e_i in e for e_j in e_i):
                        valid_format = True

        if valid_format:
            self._entanglement = entanglement
        else:
            raise NotImplementedError('Unsupported format, {}'.format(entanglement))

    @property
    def num_layers(self):
        """Return the number of layers in the n-local circuit."""
        return 2 * self._reps + int(not self._skip_final_rotation_layer)

    @property
    def initial_state(self) -> 'InitialState':
        """Return the initial state that is added in front of the NLocal.

        Returns:
            The initial state.
        """
        return self._initial_state

    @initial_state.setter
    def initial_state(self, initial_state: 'InitialState') -> None:
        """Set the initial state.

        Note that this sets the number of qubits to the width of the initial state.

        Args:
            initial_state: The new initial state.

        Raises:
            ValueError: If the number of qubits has been set before and the initial state has
                less qubits than this number of qubits.
        """
        # If there is an initial state object, check that the number of qubits is compatible
        # construct the circuit immediately. If the InitialState could modify the number of qubits
        # we could also do this later at circuit construction.
        self._initial_state = initial_state

        # construct the circuit of the initial state
        self._initial_state_circuit = initial_state.construct_circuit(mode='circuit')

        # the initial state dictates the number of qubits since we do not have information
        # about on which qubits the initial state acts
        if self._num_qubits is not None and \
                self._initial_state_circuit.num_qubits < self._num_qubits:
            raise ValueError('The provided initial state has less qubits than the NLocal.')

        self._num_qubits = self._initial_state_circuit.num_qubits
        self._invalidate()

    @property
    def insert_barriers(self) -> bool:
        """Check whether the NLocal inserts barriers or not.

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

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits in this NLocal.

        If the number of qubits has not been explicitly set via the setter or the initial state
        (which dictates the number of qubits), infer the number of qubits from the qubit indices
        of the blocks.

        Returns:
            The number of qubits.
        """
        # get the maximum number of qubits from the qubit indices
        return self._num_qubits or 0

    @num_qubits.setter
    def num_qubits(self, num_qubits: int) -> None:
        """Set the number of qubits for the NLocal.

        This does not change the qubit indices. If the number of qubits is not sufficient for the
        blocks in the NLocal, an err will be thrown upon circuit construction.

        Args:
            The new number of qubits.
        """
        if self._num_qubits != num_qubits:
            # invalidate the circuit
            self._invalidate()
            self._num_qubits = num_qubits

    @property
    def parameter_bounds(self) -> List[Tuple[float, float]]:
        """Parameter bounds.

        TODO change to return (-np.inf, np.inf) as unbounded?

        Returns:
            A list of pairs indicating the bounds, as (lower, upper).
            None indicates an unbounded parameter in the corresponding direction.
            If None is returned, problem is fully unbounded.
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

    @property
    def num_parameters_settable(self):
        """The number of total parameters that can be set to distinct values."""
        num = 0
        for i in range(self._reps):
            for block in self.entanglement_blocks:
                num += len(self.entangler_maps[i]) * len(get_parameters(block))
            for block in self.rotation_blocks:
                num += self.num_qubits // block.num_qubits * len(get_parameters(block))

        if not self._skip_final_rotation_layer:
            for block in self.rotation_blocks:
                num += self.num_qubits // block.num_qubits * len(get_parameters(block))

        return num

    @property
    def num_parameters(self):
        """The number of free parameters in the circuit."""
        self._build()
        return len(set(self._ordered_parameters))

    @property
    def parameters(self) -> Set[Parameter]:
        """Get the parameters of the NLocal.

        Only the so-called "surface parameters" of the NLocal are subject to change, these
        can be modified and re-assigned to new values. Below, the NLocal keeps track of the
        "base parameters" which remain unique.

        Returns:
            A list containing the surface parameters.
        """
        self._build()
        return super().parameters

    @parameters.setter
    def parameters(self, params: Union[dict, List[float], List[Parameter], ParameterVector]
                   ) -> None:
        """Set the parameters of the NLocal.

        This sets the surface parameters, not the base parameters.

        Args:
            The new parameters.

        Raises:
            ValueError: If the number of provided parameters does not match the number of
                parameters of the NLocal.
            TypeError: If the type of `params` is not supported.
        """
        self.assign_parameters(params, inplace=True)

    @property
    def reps(self) -> Union[int, List[int]]:
        """Return reps as integer, or if not available, as list.

        Returns:
            The repetitions. If it is an integer the repetitions specify how often
            all blocks are repeated. If a list of integers, each element is an index specifying
            which block is added to the NLocal.
        """
        return self._reps

    @reps.setter
    def reps(self, repetitions: int) -> None:
        """Set the repetitions.

        Args:
            repetitions: The new repetitions.
        """
        if repetitions != self._reps:
            self._invalidate()
            self._reps = repetitions

    @property
    def data(self):
        if self._data is None:
            self._build()
        return self._data

    @property
    def qregs(self):
        """The quantum registers associated with the circuit."""
        if self._data is None:
            self._build()
        return self._qregs

    @qregs.setter
    def qregs(self, qregs):
        """Set the quantum registers associated with the circuit."""
        self._qregs = qregs

    def _invalidate(self):
        """Invalidate the current circuit build."""
        self._data = None
        self._parameter_table = ParameterTable()

    def compose(self,
                other: Union['NLocal', Instruction, QuantumCircuit],
                indices: Optional[Union[List[int], str, List[List[int]]]] = None,
                front: bool = False,
                ) -> 'NLocal':
        """Append another layer to the NLocal.

        Args:
            other: The layer to compose, can be another NLocal, an Instruction or Gate,
                or a QuantumCircuit.
            indices: The qubit indices or entanglement.
            front: If True, ``other`` is appended to the front, else to the back.

        Returns:
            self, such that chained composes are possible.

        Raises:
            TypeError: If `other` is not compatible, i.e. is no Instruction and does not have a
                `to_instruction` method.
        """
        block = self._convert_to_block(other)

        if isinstance(indices, list) and not isinstance(indices[0], list):
            entanglement = [indices]
        else:
            entanglement = indices or [list(range(block.num_qubits))]

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

            block, entangler_map = self.blocks[-1], self.entangler_maps[-1]

            layer = QuantumCircuit(self.num_qubits)
            for i in entangler_map:
                params = self.ordered_parameters[-len(get_parameters(block)):]
                parametrized_block = self._parametrize_block(block, params)
                layer.append(parametrized_block.to_instruction(), i)

            self += layer
        else:
            # cannot prepend a block currently, just rebuild
            self._invalidate()

        return self

    def assign_parameters(self, param_dict: Union[dict, List[float], List[Parameter],
                                                  ParameterVector],
                          inplace: bool = False) -> QuantumCircuit:
        """Bind ``params`` to the underlying circuit of the NLocal.

        This method allows handling of both ``qiskit.circuit.Parameter`` objects and numbers.
        It returns a copy of the internally stored circuit with the new specified parameters.

        Returns:
            A copy of the NLocal circuit with the specified parameters.

        Raises:
            AttributeError: If the parameters are given as list and do not match the number
                of parameters.
        """
        if self._data is None:
            self._build()

        if not isinstance(param_dict, dict):
            if len(param_dict) != self.num_parameters:
                raise AttributeError('If the parameters are provided as list, the size must match '
                                     'the number of parameters ({}), but {} are given.'.format(
                                         self.num_parameters, len(param_dict)
                                     ))
            param_dict = dict(zip(self._ordered_parameters, param_dict))

        if inplace:
            new = [param_dict.get(param, param) for param in self.ordered_parameters]
            self._ordered_parameters = new

        return super().assign_parameters(param_dict, inplace=inplace)

    def to_circuit(self):
        """Build and return the circuit."""
        warnings.warn('Dont use to circuit!')
        self._build()
        return self

    def construct_circuit(self,
                          params: Union[List[float], List[Parameter], ParameterVector],
                          q: Optional[QuantumRegister] = None,
                          ) -> QuantumCircuit:
        """Deprecated, use `to_circuit()`.

        Args:
            params: The parameters for the NLocal.
            q: The qubit register to use to build the circuit. If None, a new register with the
                name 'q' is created.

        Returns:
            The NLocal as circuit with the specified parameters.

        Raises:
            ValueError: If the qubit register is provided but the length does not coincide with the
               number of qubits of the NLocal.
        """
        self.parameters = params
        if q is None:
            circuit = QuantumCircuit(self.num_qubits)
        elif len(q) != self.num_qubits:
            raise ValueError('The size of the register is not equal to the number of qubits.')
        else:
            circuit = QuantumCircuit(q)

        circuit += self.to_circuit()
        return circuit

    def _parametrize_block(self, block: Instruction,
                           params: Optional[List[Parameter]]) -> QuantumCircuit:
        """Convert ``block`` to a circuit of correct width and parameterized with the
        specified parameters.

        Args:
            block: The instruction to which the base parameters are bound.
            params: The parameters to bind to the block.

        Returns:
            The block as circuit of width ``self.num_qubits`` where the parameters have been
            substituted as specified.
        """
        circuit = QuantumCircuit(block.num_qubits)
        circuit.append(block, list(range(block.num_qubits)))
        if params is not None and self._overwrite_block_parameters:
            update = dict(zip(circuit.parameters, params))
            circuit = circuit.assign_parameters(update, inplace=False)

        return circuit

    def _build_rotation_layer(self, param_iter, i):
        """Build a rotation layer."""
        # rotation layer
        for block in self.rotation_blocks:
            layer = QuantumCircuit(*self.qregs)
            block_indices = [
                list(range(j * block.num_qubits, (j + 1) * block.num_qubits))
                for j in range(self.num_qubits // block.num_qubits)
            ]
            for indices in block_indices:
                params = [next(param_iter) for _ in range(len(get_parameters(block)))]
                parametrized_block = self._parametrize_block(block, params)
                layer.append(parametrized_block, indices)

            self += layer

    def _build_entanglement_layer(self, param_iter, i):
        """Build an entanglement layer."""
        # entanglement layer
        for j, block in enumerate(self.entanglement_blocks):
            layer = QuantumCircuit(*self.qregs)
            entangler_map = self.get_entangler_map(i, j, block.num_qubits)

            for indices in entangler_map:
                params = [next(param_iter) for _ in range(len(get_parameters(block)))]
                parametrized_block = self._parametrize_block(block, params)
                layer.append(parametrized_block, indices)

            self += layer

    def _build_additional_layers(self, which):
        if which == 'appended':
            blocks = self._appended_blocks
            entanglements = self._appended_entanglement
        elif which == 'prepended':
            blocks = reversed(self._prepended_blocks)
            entanglements = reversed(self._prepended_entanglement)
        else:
            raise ValueError('`which` must be either `appended` or `prepended`.')

        for i, (block, ent) in enumerate(zip(blocks, entanglements)):
            layer = QuantumCircuit(*self.qregs)
            if isinstance(ent, str):
                ent = get_entangler_map(block.num_block_qubits, self.num_qubits, ent)
            for indices in ent:
                layer.append(block, indices)
            self += layer

    def _build(self) -> None:
        """Build the circuit."""
        if self._data:
            return

        _ = self._configuration_is_valid()

        self._data = []

        if self.num_qubits == 0:
            return

        # use the initial state circuit if it is not None
        if self._initial_state:
            circuit = self._initial_state.construct_circuit('circuit')
            self += circuit
        else:
            self.qregs = [QuantumRegister(self.num_qubits, name='q')]
            # circuit = QuantumCircuit(q)
        param_iter = iter(self.ordered_parameters)

        # build the prepended layers
        self._build_additional_layers('prepended')

        # main loop to build the entanglement and rotation layers
        for i in range(self.reps):
            # insert barrier if specified and there is a preceding layer
            if self._insert_barriers and (i > 0 or len(self._prepended_blocks) > 0):
                self.barrier()

            # build the rotation layer
            self._build_rotation_layer(param_iter, i)

            # barrier in between rotation and entanglement layer
            if self._insert_barriers and len(self._rotation_blocks) > 0:
                self.barrier()

            # build the entanglement layer
            self._build_entanglement_layer(param_iter, i)

        # add the final rotation layer
        if not self._skip_final_rotation_layer:
            if self.insert_barriers:
                self.barrier()
            self._build_rotation_layer(param_iter, i)

        # add the appended layers
        self._build_additional_layers('appended')


def get_parameters(block: Union[QuantumCircuit, Instruction]) -> List[Parameter]:
    """Return the list of Parameters inside block.

    TODO This functionality will be moved to another location, this is just a helper.
    """
    if isinstance(block, QuantumCircuit):
        return list(block.parameters)
    else:
        return [p for p in block.params if isinstance(p, ParameterExpression)]


def get_entangler_map(num_block_qubits: int, num_circuit_qubits: int, entanglement: str,
                      offset: int = 0) -> List[Sequence[int]]:
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
        raise ValueError('The number of block qubits must be smaller or equal to the number of '
                         'qubits in the circuit.')

    if entanglement == 'full':
        return list(combinations(list(range(n)), m))
    if entanglement in ['linear', 'circular', 'sca']:
        linear = [tuple(range(i, i + m)) for i in range(n - m + 1)]
        # if the number of block qubits is 1, we don't have to add the 'circular' part
        if entanglement == 'linear' or m == 1:
            return linear

        # circular equals linear plus top-bottom entanglement
        circular = [tuple(range(n - m + 1, n)) + (0,)] + linear
        if entanglement == 'circular':
            return circular

        # sca is circular plus shift and reverse
        shifted = circular[-offset:] + circular[:-offset]
        if offset % 2 == 1:  # if odd, reverse the qubit indices
            sca = [ind[::-1] for ind in shifted]
        else:
            sca = shifted

        return sca

    else:
        raise ValueError('Unsupported entanglement type: {}'.format(entanglement))
