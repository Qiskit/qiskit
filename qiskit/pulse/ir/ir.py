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

# pylint: disable=cyclic-import

"""
=========
Pulse IR
=========
"""

from __future__ import annotations
from collections import defaultdict

import copy

import rustworkx as rx
from rustworkx import PyDAG
from rustworkx.visualization import graphviz_draw

from qiskit.pulse.transforms import AlignmentKind
from qiskit.pulse import Instruction
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.model import PulseTarget, Frame, MixedFrame


class SequenceIR:
    """IR representation of instruction sequences

    ``SequenceIR`` is the backbone of the intermediate representation used in the Qiskit Pulse compiler.
    A pulse program is represented as a single ``SequenceIR`` object, with elements
    which include ``IrInstruction`` objects and other nested ``SequenceIR`` objects.
    """

    _InNode = object()
    _OutNode = object()

    def __init__(self, alignment: AlignmentKind):
        """Create new ``SequenceIR``

        Args:
            alignment: The alignment of the object.
        """
        self._alignment = alignment

        self._time_table = defaultdict(lambda: None)
        self._sequence = rx.PyDAG(multigraph=False)
        self._sequence.add_nodes_from([SequenceIR._InNode, SequenceIR._OutNode])

    @property
    def alignment(self) -> AlignmentKind:
        """Return the alignment of the SequenceIR"""
        return self._alignment

    @property
    def inst_targets(self) -> set[PulseTarget | Frame | MixedFrame]:
        """Recursively return a set of all Instruction.inst_target in the SequenceIR"""
        inst_targets = set()
        for elm in self.elements():
            if isinstance(elm, SequenceIR):
                inst_targets |= elm.inst_targets
            else:
                inst_targets.add(elm.inst_target)
        return inst_targets

    @property
    def sequence(self) -> PyDAG:
        """Return the DAG sequence of the SequenceIR"""
        return self._sequence

    @property
    def time_table(self) -> defaultdict:
        """Return the timetable of the SequenceIR"""
        return self._time_table

    def append(self, element: SequenceIR | Instruction) -> int:
        """Append element to the SequenceIR

        Args:
            element: The element to be added, either a pulse ``Instruction`` or ``SequenceIR``.

        Returns:
            The index of the added element in the sequence.
        """
        return self._sequence.add_node(element)

    def elements(self) -> list[SequenceIR | Instruction]:
        """Return a list of all elements in the SequenceIR"""
        return self._sequence.nodes()[2:]

    def scheduled_elements(
        self, recursive: bool = False
    ) -> list[tuple[int | None, SequenceIR | Instruction]]:
        """Return a list of scheduled elements.

        Args:
            recursive: Boolean flag. If ``False``, returns only immediate children of the object
                (even if they are of type ``SequenceIR``). If ``True``, recursively returns only
                instructions, if all children ``SequenceIR`` are scheduled. Defaults to ``False``.

        Returns:
            A list of tuples of the form (initial_time, element). if all elements are scheduled,
            the returned list will be sorted in ascending order, according to initial time.
        """
        if recursive:
            listed_elements = self._recursive_scheduled_elements(0)
        else:
            listed_elements = [
                (self._time_table[ind], self._sequence.get_node_data(ind))
                for ind in self._sequence.node_indices()
                if ind not in (0, 1)
            ]

        if all(x[0] is not None for x in listed_elements):
            listed_elements.sort(key=lambda x: x[0])
        return listed_elements

    def _recursive_scheduled_elements(
        self, time_offset: int
    ) -> list[tuple[int | None, SequenceIR | Instruction]]:
        """Helper function to recursively return the scheduled elements.

        The absolute timing is tracked via the `time_offset`` argument, which represents the initial time
        of the block itself.

        Args:
            time_offset: The initial time of the ``SequenceIR`` object itself.

        Raises:
            PulseError: If children ``SequenceIR`` objects are not scheduled.
        """
        scheduled_elements = []
        for ind in self._sequence.node_indices():
            if ind not in [0, 1]:
                node = self._sequence.get_node_data(ind)
                try:
                    time = self._time_table[ind] + time_offset
                except TypeError as ex:
                    raise PulseError(
                        "Can not return recursive list of scheduled elements"
                        " if sub blocks are not scheduled."
                    ) from ex

                if isinstance(node, SequenceIR):
                    scheduled_elements.extend(node._recursive_scheduled_elements(time))
                else:
                    scheduled_elements.append((time, node))

        return scheduled_elements

    def initial_time(self) -> int | None:
        """Return initial time.

        Defaults to ``None``.
        """
        first_nodes = self._sequence.successor_indices(0)
        if not first_nodes:
            return None
        try:
            return min(self._time_table[ind] for ind in first_nodes)
        except TypeError:
            return None

    def final_time(self) -> int | None:
        """Return final time.

        Defaults to ``None``.
        """
        last_nodes = self._sequence.predecessor_indices(1)
        if not last_nodes:
            return None
        try:
            return max(
                self._time_table[ind] + self._sequence.get_node_data(ind).duration
                for ind in last_nodes
            )
        except TypeError:
            return None

    @property
    def duration(self) -> int | None:
        """Return the duration of the SequenceIR.

        Defaults to ``None``.
        """
        try:
            return self.final_time() - self.initial_time()
        except TypeError:
            return None

    def draw(self):
        """Draw the graph of the SequenceIR"""

        def _draw_nodes(n):
            if n is SequenceIR._InNode or n is SequenceIR._OutNode:
                return {"fillcolor": "grey", "style": "filled"}
            try:
                name = " " + n.name
            except (TypeError, AttributeError):
                name = ""
            return {"label": f"{n.__class__.__name__}" + name}

        return graphviz_draw(
            self.sequence,
            node_attr_fn=_draw_nodes,
        )

    def copy(self) -> SequenceIR:
        """Create a copy of ``SequenceIR``.

        The returned copy can be safely mutated without affecting the original object, while immutable
        objects are still passed as reference for memory efficiency.

        .. warning::
            If an :class:`.qiskit.pulse.Instruction` instance attached to the node of the
            internal graph is modified via :meth:`.sequence` or :meth:`.elements`, it will
            also modify the data in the original object.
            Although this is not an expected usage of ``SequenceIR``,
            deepcopy is also available at the price of performance
            to protect these node data from any modification.

        ``SequenceIR`` is poorly suited for both shallow and deep copy. A shallow copy
        will contain references to mutable properties like ``sequence`` and ``time_table``.
        A deep copy on the other hand will needlessly copy immutable objects like
        :class:`.qiskit.pulse.Instruction`. This function returns a "semi-deep" copy -
        A new object containing new objects for ``sequence`` and ``time_table``. However,
        node data of type :class:`.qiskit.pulse.Instruction` will be passed as a reference.
        Nested ``SequenceIR`` objects are copied using the same logic.

        Returns: A copy of the object.
        """
        copied = object.__new__(self.__class__)
        copied._alignment = self.alignment
        copied._time_table = copy.copy(self._time_table)
        copied._sequence = copy.copy(self._sequence)
        for node_index in copied.sequence.node_indices():
            if isinstance(nested := copied._sequence[node_index], SequenceIR):
                copied._sequence[node_index] = nested.copy()

        return copied

    def __eq__(self, other: SequenceIR):
        if other.alignment != self.alignment:
            return False
        return rx.is_isomorphic_node_match(self._sequence, other._sequence, lambda x, y: x == y)

        # TODO : What about the time_table? The isomorphic comparison allows for the indices
        #  to be different, But then it's not straightforward to compare the time_table.
        #  It is reasonable to assume that blocks with the same alignment and the same sequence
        #  will result in the same time_table, but this decision should be made consciously.

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}
        copied = self.__class__(copy.deepcopy(self.alignment, memo=memo))
        memo[id(self)] = copied
        copied._time_table = copy.copy(self._time_table)  # Only int keys and values.
        copied._sequence = copy.deepcopy(self._sequence, memo=memo)
        # To ensure that copied object will be equal to the original.
        copied._sequence[0] = self._InNode
        copied._sequence[1] = self._OutNode
        return copied

    # TODO : __repr__
