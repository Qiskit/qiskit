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

from qiskit import QiskitError


class Layout(dict):
    """ Two-ways dict to represent a Layout."""

    def __init__(self, input_=None):
        dict.__init__(self)
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
        for key, value in input_dict.items():
            self[key] = value

    def from_list(self, input_list):
        """
        Populates a Layout from a list.

        Args:
            input_list (list): For example,
            [(QuantumRegister(3, 'qr'), 0), None,
             (QuantumRegister(3, 'qr'), 2), (QuantumRegister(3, 'qr'), 3)]
        """
        for key, value in enumerate(input_list):
            self[key] = value

    def __getitem__(self, item):
        if isinstance(item, int) and item < len(self) and item not in self:
            return None
        return dict.__getitem__(self, item)

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        if value in self:
            del self[value]
        if key is not None:
            dict.__setitem__(self, key, value)
        if value is not None:
            dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        return max([key for key in self.keys() if isinstance(key, int)], default=-1) + 1

    # Override dict's built-in copy method which would return a dict instead of a Layout.
    def copy(self):
        return type(self)(self)

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
            physical_bit = len(self)
        self[virtual_bit] = physical_bit

    def add_register(self, reg):
        """
        Adds at the end physical_qubits that map each bit in reg.
        Args:
            reg (Register): A (qu)bit Register. For example, QuantumRegister(3, 'qr').
        """
        for bit in reg:
            self.add(bit)

    def set_length(self, amount_of_physical_bits):
        """
        Extends the layout length to `amount_of_physical_bits`.
        Args:
            amount_of_physical_bits (int): The amount of physical_qubits to
            set in the layout.
        Raises:
            LayoutError: If amount_of_physical_bits is used to reduced the
            length instead of extending it.
        """
        current_length = len(self)
        if amount_of_physical_bits < current_length:
            raise LayoutError('Lenght setting cannot be smaller than the current amount of physical'
                              ' (qu)bits.')
        for new_physical_bit in range(current_length, amount_of_physical_bits):
            self[new_physical_bit] = None

    def idle_physical_bits(self):
        """
        Returns a list of physical (qu)bits that are not mapped to a virtual (qu)bit.
        """
        idle_physical_bit_list = []
        for physical_bit in range(self.__len__()):
            if self[physical_bit] is None:
                idle_physical_bit_list.append(physical_bit)
        return idle_physical_bit_list

    def get_virtual_bits(self):
        """
        Returns the dictionary where the keys are virtual (qu)bits and the
        values are physical (qu)bits.
        """
        return {key: value for key, value in self.items() if isinstance(key, tuple)}

    def get_physical_bits(self):
        """
        Returns the dictionary where the keys are physical (qu)bits and the
        values are virtual (qu)bits.
        """
        return {key: value for key, value in self.items() if isinstance(key, int)}

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
            if physical not in another_layout:
                raise LayoutError('The wire_map_from_layouts() method does not support when the'
                                  ' other layout (another_layout) is smaller.')
            edge_map[virtual] = another_layout[physical]

        return edge_map


class LayoutError(QiskitError):
    """Errors raised by the layout object."""

    def __init__(self, *msg):
        """Set the error message."""
        super().__init__(*msg)
        self.msg = ' '.join(msg)

    def __str__(self):
        """Return the message."""
        return repr(self.msg)
