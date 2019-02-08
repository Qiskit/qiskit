# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
A two-ways dict that represent a layout.

Layout is the relation between virtual (qu)bits and physical (qu)bits.
Virtual (qu)bits are tuples (eg, `(QuantumRegister(3, 'qr'),2)`.
Physical (qu)bits are numbers.
"""
import warnings

from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.mapper.exceptions import LayoutError
from qiskit.circuit.register import Register


class Layout():
    """ Two-ways dict to represent a Layout."""

    def __init__(self, input_=None):
        self._p2v = {}
        self._v2p = {}
        if isinstance(input_, dict):
            self.from_dict(input_)
        if isinstance(input_, list):
            self.from_list(input_)

    def from_dict(self, input_dict):
        """
        Populates a Layout from a dictionary.

        Args:
            input_dict (dict): For example,
            {(QuantumRegister(3, 'qr'), 0): 0,
             (QuantumRegister(3, 'qr'), 1): 1,
             (QuantumRegister(3, 'qr'), 2): 2}
        """

        # TODO (luciano): Remove this full block after 0.8.
        #  its here to support {("qr", 0): ("q", 0),...}
        if len(input_dict) >= 1:
            key = list(input_dict.keys())[0]
            value = input_dict[key]
            if (isinstance(key, tuple) and  # pylint: disable=too-many-boolean-expressions
                    len(key) == 2 and
                    isinstance(key[0], str) and
                    isinstance(key[1], int) and
                    isinstance(value, tuple) and
                    len(value) == 2 and
                    isinstance(key[0], str) and
                    isinstance(key[1], int)):
                warnings.warn("This form of dictionary (i.e. {(\"%s\",%s):(\"%s\",%s), ...} is "
                              "going to be deprecated after 0.8." % (key + value),
                              DeprecationWarning)
                qreg_names = {qubit[0] for qubit in input_dict.keys()}
                qregs = {}
                for qreg_name in qreg_names:
                    qregs[qreg_name] = QuantumRegister(
                        max([qubit[1] for qubit in input_dict.keys() if qubit[0] == qreg_name]) + 1,
                        qreg_name)
                new_input_dict = {}
                for key, value in input_dict.items():
                    new_input_dict[value[1]] = (qregs[key[0]], key[1])
                input_dict = new_input_dict

        for key, value in input_dict.items():
            virtual, physical = Layout.order_based_on_type(key, value)
            self._v2p[virtual] = physical
            self._p2v[physical] = virtual

    def from_list(self, input_list):
        """
        Populates a Layout from a list.

        Args:
            input_list (list): For example,
            [(QuantumRegister(3, 'qr'), 0), None,
             (QuantumRegister(3, 'qr'), 2), (QuantumRegister(3, 'qr'), 3)]

        Raises:
            LayoutError: If the elements are not (Register, integer) or None
        """
        for physical, virtual in enumerate(input_list):
            if Layout.is_virtual(virtual):
                self._set_type_checked_item(virtual, physical)
            else:
                raise LayoutError("The list should contain elements of the form"
                                  " (Register, integer) or None")

    @staticmethod
    def order_based_on_type(value1, value2):
        """decides which one is physical/virtual based on the type. Returns (virtual, physical)"""
        if isinstance(value1, int) and Layout.is_virtual(value2):
            physical = value1
            virtual = value2
        elif isinstance(value2, int) and Layout.is_virtual(value1):
            physical = value2
            virtual = value1
        else:
            raise LayoutError('The map (%s -> %s) has to be a ((Register, integer) -> integer)'
                              ' or the other way around.' % (type(value1), type(value2)))
        return virtual, physical

    @staticmethod
    def is_virtual(value):
        """Checks if value has the format of a virtual qubit """
        return value is None or isinstance(value, tuple) and len(value) == 2 and isinstance(
            value[0], Register) and isinstance(value[1], int)

    def __getitem__(self, item):
        if item in self._p2v:
            return self._p2v[item]
        if item in self._v2p:
            return self._v2p[item]
        raise KeyError('The item %s does not exist in the Layout' % (item,))

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
            del self._p2v[key]
            del self._v2p[self._p2v[key]]
        elif isinstance(key, tuple) and \
                len(key) == 2 and \
                isinstance(key[0], Register) and isinstance(key[1], int):
            del self._v2p[key]
            del self._p2v[self._v2p[key]]
        else:
            raise LayoutError('The key to remove should be of the form'
                              ' (Register, integer) or integer) and %s was provided' % (type(key),))

    def __len__(self):
        return len(self._p2v)

    def copy(self):
        """Returns a copy of a Layout instance."""
        layout_copy = type(self)()

        layout_copy._p2v = self._p2v.copy()
        layout_copy._v2p = self._v2p.copy()

        return layout_copy

    def add(self, virtual_bit, physical_bit=None):
        """
        Adds a map element between `bit` and `physical_bit`. If `physical_bit` is not
        defined, `bit` will be mapped to a new physical bit (extending the length of the
        layout by one.)
        Args:
            virtual_bit (tuple): A (qu)bit. For example, (QuantumRegister(3, 'qr'),2).
            physical_bit (int): A physical bit. For example, 3.
        """
        if physical_bit is None:
            physical_candidate = len(self)
            while physical_candidate in self._p2v:
                physical_candidate += 1
            physical_bit = physical_candidate
        self[virtual_bit] = physical_bit

    def add_register(self, reg):
        """
        Adds at the end physical_qubits that map each bit in reg.
        Args:
            reg (Register): A (qu)bit Register. For example, QuantumRegister(3, 'qr').
        """
        for bit in reg:
            self.add(bit)

    def get_registers(self):
        """
        Returns the registers in the layout [QuantumRegister(2, 'qr0'), QuantumRegister(3, 'qr1')]
        Returns:
            List: A list of Register in the layout
        """
        return list(self.get_virtual_bits().keys())

    def idle_physical_bits(self):
        """
        Returns a list of physical (qu)bits that are not mapped to a virtual (qu)bit.
        """
        idle_physical_bit_list = []
        for physical_bit in self.get_physical_bits():
            if self._p2v[physical_bit] is None:
                idle_physical_bit_list.append(physical_bit)
        return idle_physical_bit_list

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
        """ Swaps the map between left and right.
        Args:
            left (tuple or int): Item to swap with right.
            right (tuple or int): Item to swap with left.
        Raises:
            LayoutError: If left and right have not the same type.
        """
        if type(left) is not type(right):
            raise LayoutError('The method swap only works with elements of the same type.')
        temp = self[left]
        self[left] = self[right]
        self[right] = temp

    def combine_into_edge_map(self, another_layout):
        """ Combines self and another_layout into an "edge map".

        For example::

              self       another_layout  resulting edge map
           qr_1 -> 0        0 <- q_2         qr_1 -> q_2
           qr_2 -> 2        2 <- q_1         qr_2 -> q_1
           qr_3 -> 3        3 <- q_0         qr_3 -> q_0

        The edge map is used to compose dags via, for example, compose_back.

        Args:
            another_layout (Layout): The other layout to combine.
        Returns:
            dict: A "edge map".
        Raises:
            LayoutError: another_layout can be bigger than self, but not smaller. Otherwise, raises.
        """
        edge_map = dict()

        for virtual, physical in self.get_virtual_bits().items():
            if physical not in another_layout._p2v:
                raise LayoutError('The wire_map_from_layouts() method does not support when the'
                                  ' other layout (another_layout) is smaller.')
            edge_map[virtual] = another_layout[physical]

        return edge_map

    @staticmethod
    def generate_trivial_layout(*regs):
        """
        Creates a trivial ("one-to-one") Layout with the registers in `regs`.
        Args:
            *regs (Registers): registers to include in the layout.
        Returns:
            Layout: A layout with all the `regs` in the given order.
        """
        layout = Layout()
        for reg in regs:
            layout.add_register(reg)
        return layout
