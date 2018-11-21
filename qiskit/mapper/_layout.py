# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
A two-ways dict that represent a layout.

Layout is the relation between (qu)bits and wires.
(Qu)Bits are tuples (eg, `(QuantumRegister(3, 'qr'),2)`.
Wires are numbers.
"""

from qiskit import QISKitError


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

    def add(self, bit, wire=None):
        """
        Adds a map element between `bit` and `wire`. If `wire` is not defined, `bit`
        will be mapped to a new wire (extending the length of the layout by one.)
        Args:
            bit (tuple): A (qu)bit. For example, (QuantumRegister(3, 'qr'),2).
            wire (int): A wire. For example, 3.
        """
        if wire is None:
            wire = len(self)
        self[bit] = wire

    def add_register(self, reg):
        """
        Adds at the end wires that map each bit in reg.
        Args:
            reg (Register): A (qu)bit Register. For example, QuantumRegister(3, 'qr').
        """
        for bit in reg:
            self.add(bit)

    def set_length(self, amount_of_wires):
        """
        Extends the layout length to `amount_of_wires`.
        Args:
            amount_of_wires (int): The amount of wires to set in the layout.
        Raises:
            LayoutError: If amount_of_wires is used to reduced the length instead
                of extending it.
        """
        current_length = len(self)
        if amount_of_wires < current_length:
            raise LayoutError('Lenght setting cannot be smaller than the current amount of wires.')
        for new_wire in range(current_length, amount_of_wires):
            self[new_wire] = None

    def idle_wires(self):
        """
        Returns the wires that are not mapped to a (qu)bit.
        """
        idle_wire_list = []
        for wire in range(self.__len__()):
            if self[wire] is None:
                idle_wire_list.append(wire)
        return idle_wire_list

    def get_bits(self):
        """
        Returns the dictionary where the keys are (qu)bits and the
        values are wires.
        """
        return {key: value for key, value in self.items() if isinstance(key, tuple)}

    def get_wires(self):
        """
        Returns the dictionary where the keys are wires and the
        values are (qu)bits.
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


class LayoutError(QISKitError):
    """Errors raised by the layout object."""
    def __init__(self, *msg):
        """Set the error message."""
        super().__init__(*msg)
        self.msg = ' '.join(msg)

    def __str__(self):
        """Return the message."""
        return repr(self.msg)
