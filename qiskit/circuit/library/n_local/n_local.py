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
# import warnings
import logging
from typing import Union, Optional, List, Any, Tuple, Sequence, Set
from itertools import combinations

import numpy
from qiskit import QuantumCircuit, transpile, QuantumRegister
from qiskit.circuit import Instruction, Parameter, ParameterVector, ParameterExpression

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
                 reps: int = 3,
                 insert_barriers: bool = False,
                 parameter_prefix: str = 'θ',
                 overwrite_block_parameters: Union[bool, List[List[Parameter]]] = True,
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
        self.entanglement_blocks = entanglement_blocks or []
        self._rotation_blocks = []
        self.rotation_blocks = rotation_blocks or []
        self._ordered_parameters = ParameterVector(name=parameter_prefix)
        self._overwrite_block_parameters = overwrite_block_parameters

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

    # def __iadd__(self, other: Union['NLocal', Instruction, QuantumCircuit]) -> 'NLocal':
    #     """Overloading += for convenience.

    #     This presumes list(range(other.num_qubits)) as qubit indices and calls self.compose().

    #     Args:
    #         other: The object to compose.

    #     Raises:
    #         TypeError: If the added type is unsupported.

    #     Returns:
    #         self
    #     """
    #     return self.compose(other)

    # def __add__(self, other: Union['NLocal', Instruction, QuantumCircuit]) -> 'NLocal':
    #     """Overloading += for convenience.

    #     This presumes list(range(other.num_qubits)) as qubit indices and calls self.compose().

    #     Args:
    #         other: The object to compose.

    #     Raises:
    #         TypeError: If the added type is unsupported.

    #     Returns:
    #         A copy of self with the other object composeed.
    #     """
    #     target = copy.deepcopy(self)
    #     return target.compose(other)

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
        # check no needed parameters are None
        if self.entanglement_blocks is None and self.rotation_blocks is None:
            raise ValueError('The blocks are not set.')

        # check the compatibility of the attributes
        if len(self.entangler_maps) != len(self._entanglement_reps()):
            raise ValueError('The number of qubit indices does not match the number of '
                             'repetitions.')

        if isinstance(self._reps, list):
            if len(self._reps) > 0:
                if max(self._reps) >= len(self._blocks):
                    raise ValueError('Trying to add a non-existing block to the circuit.')

            # if self._blockwise_base_params:
            #     if len(self._reps) != len(self._blockwise_base_params):
            #         raise ValueError('The number of repetitions ({}) does '.format(len(self.reps))
            #                          + 'not match with the number of block parameters '
            #                          + '({})'.format(len(self._blockwise_base_params)))

        if self._num_qubits:
            for layer in self.entangler_maps:
                for block in layer:
                    if max(block) >= self._num_qubits:
                        raise ValueError('The manually set number of qubits is too small for the '
                                         + 'blocks in the circuit.')

        return True

    def _reps_as_list(self) -> List[int]:
        """Return the indices of the blocks that go in the circuit.

        Returns:
            If ``reps`` has been set to a list, return that list. Otherwise, i.e. if ``reps`` is an
            integer, return ``reps * list(range(num. of blocks))``.
        """
        if isinstance(self._reps, int):
            if self.entanglement_blocks is None and self.rotation_blocks is None:
                return []
            return self._reps * list(range(len(self.blocks)))
        return self._reps

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

    @staticmethod
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
    def ordered_parameters(self) -> ParameterVector:
        """The parameters used in the underlying circuit.

        Returns:
            The parameters objects used in the circuit.
        """
        self._ordered_parameters.resize(self.num_parameters)
        return list(self._ordered_parameters)

    @ordered_parameters.setter
    def ordered_parameters(self, parameters: ParameterVector) -> None:
        """Set the parameters used in the underlying circuit.

        Args:
            The parameters to be used in the underlying circuit.
        """
        if self._circuit:
            self._circuit.assign_parameters(dict(zip(self._ordered_parameters, parameters)),
                                            inplace=True)
        self._ordered_parameters = parameters

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

        self._data = None
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

        self._data = None
        self._entanglement_blocks = [self._convert_to_block(block) for block in blocks]

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
                    self.get_entangler_map(block.num_qubits, self.num_qubits, entanglement, num)
                ]
            return entangler_maps

        if callable(entanglement):
            entangler_maps = []
            for num, i in enumerate(self._entanglement_reps()):
                ent = entanglement(num)
                block = self.entanglement_blocks[i]
                if isinstance(entanglement, str):
                    entangler_maps += [
                        self.get_entangler_map(block.num_qubits, self.num_qubits, ent, num)
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
                        self.get_entangler_map(block.num_qubits, self.num_qubits, ent, num)
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
        return len(self._entanglement_reps() + self._rotation_reps())

    @entangler_maps.setter
    def entangler_maps(self, indices: List[List[int]]) -> None:
        """Set the qubit indices per block.

        Args:
            The qubit indices specifying on which qubits each block acts on.
        """
        self._entangler_maps = indices

        # TODO check if indices is the same if s, only then invalidate the definition
        # but this setter is probably not really used anywhere except the initializer anyways
        self._data = None

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
        self._data = None

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
            self._data = None
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
        if self._num_qubits is None:
            if len(self.entangler_maps) > 0:
                flattened_indices = [i for layer in self.entangler_maps
                                     for block in layer
                                     for i in block]
                return 1 + max(flattened_indices)
            return 0

        return int(self._num_qubits)

    @num_qubits.setter
    def num_qubits(self, num_qubits: int) -> None:
        """Set the number of qubits for the NLocal.

        This does not change the qubit indices. If the number of qubits is not sufficient for the
        blocks in the NLocal, an error will be thrown upon circuit construction.

        Args:
            The new number of qubits.
        """
        if self._num_qubits != num_qubits:
            # invalidate the circuit
            self._data = None
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
        if self._circuit is None:
            _ = self.to_circuit()
        return self._bounds

    @parameter_bounds.setter
    def parameter_bounds(self, bounds: List[Tuple[float, float]]) -> None:
        """Set the parameter bounds.

        Args:
            bounds: The new parameter bounds.
        """
        self._bounds = bounds

    @property
    def num_parameters(self):
        """The number of free parameters in the circuit."""
        num = 0
        for i in self._entanglement_reps():
            num += len(self.entangler_maps[i]) * len(get_parameters(self.entanglement_blocks[i]))
        for i in self._rotation_reps():
            block = self.rotation_blocks[i]
            num += len(get_parameters(block)) * self.num_qubits // block.num_qubits
        return num

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
        return

        # TODO figure out whether it is more efficient to iterate over the list and check for
        # values in the dictionary, or iterate over the dictionary and find the according value
        # in the list. Random access via element should be much faster in the dictionary, probably.
        # if isinstance(params, dict):
        #     new_params = []
        #     for i, current_param in enumerate(self.parameters):
        #         # try to get the new value, if there is none, use the current value
        #         new_params[i] = params.get(current_param, self.parameters[i])
        #     self._surface_params = new_params

        # # if a list is provided, just assign if the sizes match
        # else:
        #     if len(params) != self.num_parameters:
        #         raise ValueError('Mismatching number of parameters! '
        #                          'Provided: {}, required: {}'
        #                          ''.format(len(params), self.num_parameters))
        #     self._surface_params = params

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
    def reps(self, repetitions: Union[int, List[int]]) -> None:
        """Set the repetitions.

        Args:
            repetitions: The new repetitions.
        """

        # TODO invalidate circuit only when reps changed
        self._data = None
        self._reps = repetitions

    def _get_default_parameters(self, start: int, num: int) -> List[Parameter]:
        """Get ``num`` default parameters. Returns the same instances if called repeatedly.

        Args:
            start: The first index of the parameters.
            num: The number of required parameters.

        Returns:
            A list of ``num`` parameters, named ``self._parameter_prefix + i``, where the index
            ``i`` runs from ``start`` to ``start + num``.

        Note:
            This method guarantees to return the same instances if the same indices are
            requested, i.e.

        Example:
            >>> self._get_default_parameters(10, 2)[0] is self._get_default_parameters(10, 2)[0]
            True

        TODO:
            Implement this more efficiently such that only the asked for params are created.
            E.g. ``_get_default_parameters(1000, 1)`` should not create 1001 parameters, but 1.
        """
        num_default_parameters = len(self._default_parameters)
        if num_default_parameters < start + num:
            self._default_parameters += [
                Parameter('{}{}'.format(self._parameter_prefix, num_default_parameters + i))
                for i in range(start + num - num_default_parameters)
            ]

        return self._default_parameters[start:start + num]

    @property
    def data(self):
        self._build()
        return self._data

    def compose(self,
                other: Union['NLocal', Instruction, QuantumCircuit],
                entangler_maps: Optional[List[int]] = None
                ) -> 'NLocal':
        """Append another layer to the NLocal.

        Args:
            other: The layer to compose, can be another NLocal, an Instruction or Gate,
                or a QuantumCircuit.
            entangler_maps: The qubit indices where to compose the layer to.
                Defaults to the first `n` qubits, where `n` is the number of qubits the layer acts
                on.

        Returns:
            self, such that chained composes are possible.

        Raises:
            TypeError: If `other` is not compatible, i.e. is no Instruction and does not have a
                `to_instruction` method.
        """
        block = self._convert_to_block(other)

        # define the the qubit indices
        if entangler_maps:
            self._entanglement = self.entangler_maps + [[entangler_maps]]
            num_qubits = max(entangler_maps)
        else:
            num_qubits = block.num_qubits

        # Convert to a list, so that if reps was an integer, the composeed block gets added
        # once and not multiple times. Must happen before composeing block to self._blocks.
        self._reps = self._reps_as_list() + [len(self._blocks)]

        # add other to the list of blocks
        self._blocks += [block]

        # We can have two cases: the composeed block fits onto the current NLocal (i.e. has
        # less of equal number of qubits), or exceeds the number of qubits.
        # In the latter case we have to add an according offset to the qubit indices.
        # Since we cannot compose a circuit of larger size to an existing circuit we have to rebuild
        if num_qubits != self.num_qubits:
            self._data = None  # rebuild circuit

        # modify the circuit accordingly
        if self._data:
            if self._insert_barriers and len(self._reps_as_list()) > 1:
                self.barrier()

            block, entangler_map = self.blocks[-1], self.entangler_maps[-1]

            layer = QuantumCircuit(self.num_qubits)
            for indices in entangler_map:
                params = self.ordered_parameters[-len(get_parameters(block)):]
                parametrized_block = self._parametrize_block(block, params)
                layer.append(parametrized_block.to_instruction(), indices)

            self += layer

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
            TypeError: If ``params`` contains an unsupported type.
        """
        self._build()
        if not isinstance(param_dict, dict):
            param_dict = dict(zip(self.ordered_parameters, param_dict))

        print('ordered', self.ordered_parameters)
        print('self', self.parameters)
        print('param_dict', param_dict)
        return super().assign_parameters(param_dict, inplace=inplace)

    def to_circuit(self):
        """Build and return the circuit."""
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

    def draw(self, *args, **kwargs):
        self._build()
        return super().draw(*args, **kwargs)

    def _build(self) -> None:
        """Build the circuit."""
        if self._data is None and self._configuration_is_valid():
            self._data = []
            self._parameter_table = {}
            if self.num_qubits > 0:
                # use the initial state circuit if it is not None
                if self._initial_state:
                    circuit = self._initial_state.construct_circuit('circuit')
                    self += circuit
                else:
                    self.qregs = [QuantumRegister(self.num_qubits, name='q')]
                    # circuit = QuantumCircuit(q)
            else:
                self.qregs = []

            # add the blocks, if they are specified
            rotation_reps, entanglement_reps = self._rotation_reps(), self._entanglement_reps()
            if len(rotation_reps) > 0 or len(entanglement_reps) > 0:
                param_iter = iter(self.ordered_parameters)

                for i in range(self.reps):

                    if self._insert_barriers and i > 0:
                        self.barrier()

                    # rotation layer
                    if len(rotation_reps) > 0:
                        block = self.rotation_blocks[rotation_reps[i]]

                        layer = QuantumCircuit(self.num_qubits)
                        block_indices = [
                            list(range(j * block.num_qubits, (j + 1) * block.num_qubits))
                            for j in range(self.num_qubits // block.num_qubits)
                        ]
                        for indices in block_indices:
                            params = [next(param_iter) for _ in range(len(get_parameters(block)))]
                            parametrized_block = self._parametrize_block(block, params)
                            layer.append(parametrized_block, indices)

                        self += layer

                    # entanglement layer
                    if len(entanglement_reps) > 0:
                        block = self.entanglement_blocks[entanglement_reps[i]]
                        entangler_map = self.entangler_maps[i]

                        layer = QuantumCircuit(self.num_qubits)
                        for indices in entangler_map:
                            params = [next(param_iter) for _ in range(len(get_parameters(block)))]
                            parametrized_block = self._parametrize_block(block, params)
                            layer.append(parametrized_block, indices)

                        self += layer

            if len(rotation_reps) > self._reps:
                if self.insert_barriers:
                    self.barrier()
                block = self.rotation_blocks[rotation_reps[self._reps]]
                layer = QuantumCircuit(self.num_qubits)
                block_indices = [
                    list(range(j * block.num_qubits, (j + 1) * block.num_qubits))
                    for j in range(self.num_qubits // block.num_qubits)
                ]
                for indices in block_indices:
                    params = [next(param_iter) for _ in range(len(get_parameters(block)))]
                    parametrized_block = self._parametrize_block(block, params)
                    layer.append(parametrized_block, indices)

                self += layer

            # store the data
            # self.qregs = circuit.qregs
            # self._data = circuit._data


def get_parameters(block: Union[QuantumCircuit, Instruction]) -> List[Parameter]:
    """Return the list of Parameters inside block.

    TODO This functionality will be moved to another location, this is just a helper.
    """
    if isinstance(block, QuantumCircuit):
        return list(block.parameters)
    else:
        return [p for p in block.params if isinstance(p, ParameterExpression)]


def combine_parameterlists(first: List[Parameter], other: List[Parameter],
                           duplicate_existing: bool = True) -> List[Parameter]:
    """Add ``other`` to the ``first`` via name, not instance.

    TODO This functionality will be moved to another location, this is just a helper.

    If the parameters in ``other`` already exists in the list, add the instance with the same
    name at the end of the list. If the name does not exist in the list, add the parameter
    in ``other``.
    This prevents having different instances with the same name in the list of parameters,
    which leads to naming conflict if a circuit is constructed with these parameters.

    Args:
        first: The list of parameters, where ``new_parameter`` is to be added.
        other: The parameter list that should be added.
        duplicate_existing: Duplicate the parameter even if it is in the list.

    Returns:
        The merged parameter list, where same names are the same instance.
    """
    for obj in [first, other]:
        if isinstance(obj, ParameterExpression):
            obj = [obj]

    for new_param in other:
        found = False
        for existing_param in first:
            if new_param.name == existing_param.name:
                if duplicate_existing:
                    first.append(existing_param)
                found = True
                break

        if not found:
            first.append(new_param)

    return first
