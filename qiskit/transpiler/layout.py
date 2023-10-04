# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
A two-ways dict to represent a layout.

Layout is the relation between virtual (qu)bits and physical (qu)bits.
Virtual (qu)bits are tuples, e.g. `(QuantumRegister(3, 'qr'), 2)` or simply `qr[2]`.
Physical (qu)bits are integers.
"""
from __future__ import annotations
from typing import List
from dataclasses import dataclass

from qiskit.circuit.quantumregister import Qubit, QuantumRegister
from qiskit.transpiler.exceptions import LayoutError
from qiskit.converters import isinstanceint


class Layout:
    """Two-ways dict to represent a Layout."""

    __slots__ = ("_regs", "_p2v", "_v2p")

    def __init__(self, input_dict=None):
        """construct a Layout from a bijective dictionary, mapping
        virtual qubits to physical qubits"""
        self._regs = []
        self._p2v = {}
        self._v2p = {}
        if input_dict is not None:
            if not isinstance(input_dict, dict):
                raise LayoutError("Layout constructor takes a dict")
            self.from_dict(input_dict)

    def __repr__(self):
        """Representation of a Layout"""
        str_list = []
        for key, val in self._p2v.items():
            str_list.append(f"{key}: {val},")
        if str_list:
            str_list[-1] = str_list[-1][:-1]
        return "Layout({\n" + "\n".join(str_list) + "\n})"

    def from_dict(self, input_dict):
        """Populates a Layout from a dictionary.

        The dictionary must be a bijective mapping between
        virtual qubits (tuple) and physical qubits (int).

        Args:
            input_dict (dict):
                e.g.::

                {(QuantumRegister(3, 'qr'), 0): 0,
                 (QuantumRegister(3, 'qr'), 1): 1,
                 (QuantumRegister(3, 'qr'), 2): 2}

                Can be written more concisely as follows:

                * virtual to physical::

                    {qr[0]: 0,
                     qr[1]: 1,
                     qr[2]: 2}

                * physical to virtual::

                    {0: qr[0],
                     1: qr[1],
                     2: qr[2]}
        """
        for key, value in input_dict.items():
            virtual, physical = Layout.order_based_on_type(key, value)
            self._p2v[physical] = virtual
            if virtual is None:
                continue
            self._v2p[virtual] = physical

    @staticmethod
    def order_based_on_type(value1, value2):
        """decides which one is physical/virtual based on the type. Returns (virtual, physical)"""
        if isinstanceint(value1) and isinstance(value2, (Qubit, type(None))):
            physical = int(value1)
            virtual = value2
        elif isinstanceint(value2) and isinstance(value1, (Qubit, type(None))):
            physical = int(value2)
            virtual = value1
        else:
            raise LayoutError(
                "The map (%s -> %s) has to be a (Bit -> integer)"
                " or the other way around." % (type(value1), type(value2))
            )
        return virtual, physical

    def __getitem__(self, item):
        if item in self._p2v:
            return self._p2v[item]
        if item in self._v2p:
            return self._v2p[item]
        raise KeyError(f"The item {item} does not exist in the Layout")

    def __contains__(self, item):
        return item in self._p2v or item in self._v2p

    def __setitem__(self, key, value):
        virtual, physical = Layout.order_based_on_type(key, value)
        self._set_type_checked_item(virtual, physical)

    def _set_type_checked_item(self, virtual, physical):
        old = self._v2p.pop(virtual, None)
        self._p2v.pop(old, None)
        old = self._p2v.pop(physical, None)
        self._v2p.pop(old, None)

        self._p2v[physical] = virtual
        if virtual is not None:
            self._v2p[virtual] = physical

    def __delitem__(self, key):
        if isinstance(key, int):
            del self._v2p[self._p2v[key]]
            del self._p2v[key]
        elif isinstance(key, Qubit):
            del self._p2v[self._v2p[key]]
            del self._v2p[key]
        else:
            raise LayoutError(
                "The key to remove should be of the form"
                " Qubit or integer) and %s was provided" % (type(key),)
            )

    def __len__(self):
        return len(self._p2v)

    def __eq__(self, other):
        if isinstance(other, Layout):
            return self._p2v == other._p2v and self._v2p == other._v2p
        return False

    def copy(self):
        """Returns a copy of a Layout instance."""
        layout_copy = type(self)()

        layout_copy._regs = self._regs.copy()
        layout_copy._p2v = self._p2v.copy()
        layout_copy._v2p = self._v2p.copy()

        return layout_copy

    def add(self, virtual_bit, physical_bit=None):
        """
        Adds a map element between `bit` and `physical_bit`. If `physical_bit` is not
        defined, `bit` will be mapped to a new physical bit.

        Args:
            virtual_bit (tuple): A (qu)bit. For example, (QuantumRegister(3, 'qr'), 2).
            physical_bit (int): A physical bit. For example, 3.
        """
        if physical_bit is None:
            if len(self._p2v) == 0:
                physical_bit = 0
            else:
                max_physical = max(self._p2v)
                # Fill any gaps in the existing bits
                for physical_candidate in range(max_physical):
                    if physical_candidate not in self._p2v:
                        physical_bit = physical_candidate
                        break
                # If there are no free bits in the allocated physical bits add new ones
                else:
                    physical_bit = max_physical + 1

        self[virtual_bit] = physical_bit

    def add_register(self, reg):
        """Adds at the end physical_qubits that map each bit in reg.

        Args:
            reg (Register): A (qu)bit Register. For example, QuantumRegister(3, 'qr').
        """
        self._regs.append(reg)
        for bit in reg:
            if bit not in self:
                self.add(bit)

    def get_registers(self):
        """
        Returns the registers in the layout [QuantumRegister(2, 'qr0'), QuantumRegister(3, 'qr1')]
        Returns:
            Set: A set of Registers in the layout
        """
        return set(self._regs)

    def get_virtual_bits(self):
        """
        Returns the dictionary where the keys are virtual (qu)bits and the
        values are physical (qu)bits.
        """
        return self._v2p

    def get_physical_bits(self):
        """
        Returns the dictionary where the keys are physical (qu)bits and the
        values are virtual (qu)bits.
        """
        return self._p2v

    def swap(self, left, right):
        """Swaps the map between left and right.

        Args:
            left (tuple or int): Item to swap with right.
            right (tuple or int): Item to swap with left.
        Raises:
            LayoutError: If left and right have not the same type.
        """
        if type(left) is not type(right):
            raise LayoutError("The method swap only works with elements of the same type.")
        temp = self[left]
        self[left] = self[right]
        self[right] = temp

    def combine_into_edge_map(self, another_layout):
        """Combines self and another_layout into an "edge map".

        For example::

              self       another_layout  resulting edge map
           qr_1 -> 0        0 <- q_2         qr_1 -> q_2
           qr_2 -> 2        2 <- q_1         qr_2 -> q_1
           qr_3 -> 3        3 <- q_0         qr_3 -> q_0

        The edge map is used to compose dags via, for example, compose.

        Args:
            another_layout (Layout): The other layout to combine.
        Returns:
            dict: A "edge map".
        Raises:
            LayoutError: another_layout can be bigger than self, but not smaller.
                Otherwise, raises.
        """
        edge_map = {}

        for virtual, physical in self._v2p.items():
            if physical not in another_layout._p2v:
                raise LayoutError(
                    "The wire_map_from_layouts() method does not support when the"
                    " other layout (another_layout) is smaller."
                )
            edge_map[virtual] = another_layout[physical]

        return edge_map

    def reorder_bits(self, bits) -> list[int]:
        """Given an ordered list of bits, reorder them according to this layout.

        The list of bits must exactly match the virtual bits in this layout.

        Args:
            bits (list[Bit]): the bits to reorder.

        Returns:
            List: ordered bits.
        """
        order = [0] * len(bits)

        # the i-th bit is now sitting in position j
        for i, v in enumerate(bits):
            j = self[v]
            order[i] = j

        return order

    @staticmethod
    def generate_trivial_layout(*regs):
        """Creates a trivial ("one-to-one") Layout with the registers and qubits in `regs`.

        Args:
            *regs (Registers, Qubits): registers and qubits to include in the layout.
        Returns:
            Layout: A layout with all the `regs` in the given order.
        """
        layout = Layout()
        for reg in regs:
            if isinstance(reg, QuantumRegister):
                layout.add_register(reg)
            else:
                layout.add(reg)
        return layout

    @staticmethod
    def from_intlist(int_list, *qregs):
        """Converts a list of integers to a Layout
        mapping virtual qubits (index of the list) to
        physical qubits (the list values).

        Args:
            int_list (list): A list of integers.
            *qregs (QuantumRegisters): The quantum registers to apply
                the layout to.
        Returns:
            Layout: The corresponding Layout object.
        Raises:
            LayoutError: Invalid input layout.
        """
        if not all(isinstanceint(i) for i in int_list):
            raise LayoutError("Expected a list of ints")
        if len(int_list) != len(set(int_list)):
            raise LayoutError("Duplicate values not permitted; Layout is bijective.")
        num_qubits = sum(reg.size for reg in qregs)
        # Check if list is too short to cover all qubits
        if len(int_list) != num_qubits:
            raise LayoutError(
                f"Integer list length ({len(int_list)}) must equal number of qubits "
                f"in circuit ({num_qubits}): {int_list}."
            )
        out = Layout()
        main_idx = 0
        for qreg in qregs:
            for idx in range(qreg.size):
                out[qreg[idx]] = int_list[main_idx]
                main_idx += 1
            out.add_register(qreg)
        if main_idx != len(int_list):
            for int_item in int_list[main_idx:]:
                out[int_item] = None
        return out

    @staticmethod
    def from_qubit_list(qubit_list, *qregs):
        """
        Populates a Layout from a list containing virtual
        qubits, Qubit or None.

        Args:
            qubit_list (list):
                e.g.: [qr[0], None, qr[2], qr[3]]
            *qregs (QuantumRegisters): The quantum registers to apply
                the layout to.
        Returns:
            Layout: the corresponding Layout object
        Raises:
            LayoutError: If the elements are not Qubit or None
        """
        out = Layout()
        for physical, virtual in enumerate(qubit_list):
            if virtual is None:
                continue
            if isinstance(virtual, Qubit):
                if virtual in out._v2p:
                    raise LayoutError("Duplicate values not permitted; Layout is bijective.")
                out[virtual] = physical
            else:
                raise LayoutError("The list should contain elements of the Bits or NoneTypes")
        for qreg in qregs:
            out.add_register(qreg)
        return out


@dataclass
class TranspileLayout:
    r"""Layout attributes from output circuit from transpiler.

    The transpiler in general is unitary-perserving up to permutations caused
    by setting and applying initial layout during the :ref:`layout_stage`
    and :class:`~.SwapGate` insertion during the :ref:`routing_stage`. To
    provide an interface to reason about these permutations caused by
    the :mod:`~qiskit.transpiler`. In general the normal interface to access
    and reason about the layout transformations made by the transpiler is to
    use the helper methods defined on this class.

    For example, looking at the initial layout, the transpiler can potentially
    remap the order of the qubits in your circuit as it fits the circuit to
    the target backend. If the input circuit was:

    .. plot:
       :include-source:

       from qiskit.circuit import QuantumCircuit, QuantumRegister

       qr = QuantumRegister(3, name="MyReg")
       qc = QuantumCircuit(qr)
       qc.h(0)
       qc.cx(0, 1)
       qc.cx(0, 2)
       qc.draw("mpl")

    Then during the layout stage the transpiler reorders the qubits to be:

    .. plot:
       :include-source:

       from qiskit import QuantumCircuit

       qc = QuantumCircuit(3)
       qc.h(2)
       qc.cx(2, 1)
       qc.cx(2, 0)
       qc.draw("mpl")

    then the output of the :meth:`.initial_virtual_layout` would be
    equivalent to::

        Layout({
            qr[0]: 2,
            qr[1]: 1,
            qr[2]: 0,
        })

    (it is also this attribute in the :meth:`.QuantumCircuit.draw` and
    :func:`.circuit_drawer` which is used to display the mapping of qubits to
    positions in circuit visualizations post-transpilation)

    Building on this above example for final layout, if the transpiler needed to
    insert swap gates during routing so the output circuit became:

    .. plot:
       :include-source:

       from qiskit import QuantumCircuit

       qc = QuantumCircuit(3)
       qc.h(2)
       qc.cx(2, 1)
       qc.swap(0, 1)
       qc.cx(2, 1)
       qc.draw("mpl")

    then the output of the :meth:`routing_permutation` method would be::

        [1, 0, 2]

    which maps the qubits at each position to their final position after any swap
    insertions caused by routing.

    There are three public attributes associated with the class, however these
    are mostly provided for backwards compatibility and represent the internal
    state from the transpiler. They are defined as:

      * :attr:`initial_layout` - This attribute is used to model the
        permutation caused by the :ref:`layout_stage` it contains a
        :class:`~.Layout` object that maps the input :class:`~.QuantumCircuit`\s
        :class:`~.Qubit` objects to the position in the output
        :class:`.QuantumCircuit.qubits` list.
      * :attr:`input_qubit_mapping` - This attribute is used to retain
        input ordering of the original :class:`~.QuantumCircuit` object. It
        maps the virtual :class:`~.Qubit` object from the original circuit
        (and :attr:`initial_layout`) to its corresponding position in
        :attr:`.QuantumCircuit.qubits` in the original circuit. This
        is needed when computing the permutation of the :class:`Operator` of
        the circuit (and used by :meth:`.Operator.from_circuit`).
      * :attr:`final_layout` - This is a :class:`~.Layout` object used to
        model the output permutation caused ny any :class:`~.SwapGate`\s
        inserted into the :class:`~.QuantumCircuit` during the
        :ref:`routing_stage`. It maps the output circuit's qubits from
        :class:`.QuantumCircuit.qubits` in the output circuit to the final
        position after routing. It is **not** a mapping from the original
        input circuit's position to the final position at the end of the
        transpiled circuit. If you need this you can use the
        :meth:`.final_index_layout` to generate this. If this is set to ``None``
        this indicates that routing was not run and it can be considered
        equivalent to a trivial layout with the qubits from the output circuit's
        :attr:`~.QuantumCircuit.qubits` list.
    """

    initial_layout: Layout
    input_qubit_mapping: dict[Qubit, int]
    final_layout: Layout | None = None
    _input_qubit_count: int | None = None
    _output_qubit_list: List[Qubit] | None = None

    def initial_virtual_layout(self, filter_ancillas: bool = False) -> Layout:
        """Return a :class:`.Layout` object for the initial layout.

        This returns a mapping of virtual :class:`~.Qubit` objects in the input
        circuit to the physical qubit selected during layout. This is analogous
        to the :attr:`.initial_layout` attribute.

        Args:
            filter_ancillas: If set to ``True`` only qubits in the input circuit
                will be in the returned layout. Any ancilla qubits added to the
                output circuit will be filtered from the returned object.
        Returns:
            A layout object mapping the input circuit's :class:`~.Qubit`
            objects to the selected physical qubits.
        """
        if not filter_ancillas:
            return self.initial_layout
        return Layout(
            {
                k: v
                for k, v in self.initial_layout.get_virtual_bits().items()
                if self.input_qubit_mapping[k] < self._input_qubit_count
            }
        )

    def initial_index_layout(self, filter_ancillas: bool = False) -> List[int]:
        """Generate an initial layout as an array of integers

        Args:
            filter_ancillas: If set to ``True`` any ancilla qubits added
                to the transpiler will not be included in the output.

        Return:
            A layout array that maps a position in the array to its new position in the output
            circuit.
        """

        virtual_map = self.initial_layout.get_virtual_bits()
        if filter_ancillas:
            output = [None] * self._input_qubit_count
        else:
            output = [None] * len(virtual_map)
        for index, (virt, phys) in enumerate(virtual_map.items()):
            if filter_ancillas and index >= self._input_qubit_count:
                break
            pos = self.input_qubit_mapping[virt]
            output[pos] = phys
        return output

    def routing_permutation(self) -> List[int]:
        """Generate a final layout as an array of integers

        If there is no :attr:`.final_layout` attribute present then that indicates
        there was no output permutation caused by routing or other transpiler
        transforms. In this case the function will return a list of ``[0, 1, 2, .., n]``
        to indicate this

        Returns:
            A layout array that maps a position in the array to its new position in the output
            circuit
        """
        if self.final_layout is None:
            return list(range(len(self._output_qubit_list)))
        virtual_map = self.final_layout.get_virtual_bits()
        return [virtual_map[virt] for virt in self._output_qubit_list]

    def final_index_layout(self, filter_ancillas: bool = True) -> List[int]:
        """Generate the final layout as an array of integers

        This method will generate an array of final positions for each qubit in the output circuit.
        For example, if you had an input circuit like::

            qc = QuantumCircuit(3)
            qc.h(0)
            qc.cx(0, 1)
            qc.cx(0, 2)

        and the output from the transpiler was::

            tqc = QuantumCircuit(3)
            qc.h(2)
            qc.cx(2, 1)
            qc.swap(0, 1)
            qc.cx(2, 1)

        then the return from this function would be a list of::

            [2, 0, 1]

        because qubit 0 in the original circuit's final state is on qubit 3 in the output circuit,
        qubit 1 in the original circuit's final state is on qubit 0, and qubit 2's final state is
        on qubit. The output list length will be as wide as the input circuit's number of qubits,
        as the output list from this method is for tracking the permutation of qubits in the
        original circuit caused by the transpiler.

        Args:
            filter_ancillas: If set to ``False`` any ancillas allocated in the output circuit will be
                included in the layout.

        Returns:
            A list of final positions for each input circuit qubit
        """
        if self._input_qubit_count is None:
            # TODO: After there is a way to differentiate the ancilla qubits added by the transpiler
            # don't use the ancilla name anymore.See #10817 for discussion on this.
            num_source_qubits = len(
                [
                    x
                    for x in self.input_qubit_mapping
                    if getattr(x, "_register", "").startswith("ancilla")
                ]
            )
        else:
            num_source_qubits = self._input_qubit_count
        if self._output_qubit_list is None:
            circuit_qubits = list(self.final_layout.get_virtual_bits())
        else:
            circuit_qubits = self._output_qubit_list

        pos_to_virt = {v: k for k, v in self.input_qubit_mapping.items()}
        qubit_indices = []
        if filter_ancillas:
            num_qubits = num_source_qubits
        else:
            num_qubits = len(self._output_qubit_list)
        for index in range(num_qubits):
            qubit_idx = self.initial_layout[pos_to_virt[index]]
            if self.final_layout is not None:
                qubit_idx = self.final_layout[circuit_qubits[qubit_idx]]
            qubit_indices.append(qubit_idx)
        return qubit_indices

    def final_virtual_layout(self, filter_ancillas: bool = True) -> Layout:
        """Generate the final layout as a :class:`.Layout` object

        This method will generate an array of final positions for each qubit in the output circuit.
        For example, if you had an input circuit like::

            qc = QuantumCircuit(3)
            qc.h(0)
            qc.cx(0, 1)
            qc.cx(0, 2)

        and the output from the transpiler was::

            tqc = QuantumCircuit(3)
            qc.h(2)
            qc.cx(2, 1)
            qc.swap(0, 1)
            qc.cx(2, 1)

        then the return from this function would be a layout object::

            Layout({
                qc.qubits[0]: 2,
                qc.qubits[1]: 0,
                qc.qubits[2]: 1,
            })

        because qubit 0 in the original circuit's final state is on qubit 3 in the output circuit,
        qubit 1 in the original circuit's final state is on qubit 0, and qubit 2's final state is
        on qubit. The output list length will be as wide as the input circuit's number of qubits,
        as the output list from this method is for tracking the permutation of qubits in the
        original circuit caused by the transpiler.

        Args:
            filter_ancillas: If set to ``False`` any ancillas allocated in the output circuit will be
                included in the layout.

        Returns:
            A layout object mapping to the final positions for each qubit
        """
        res = self.final_index_layout(filter_ancillas=filter_ancillas)
        pos_to_virt = {v: k for k, v in self.input_qubit_mapping.items()}
        return Layout({pos_to_virt[index]: phys for index, phys in enumerate(res)})
