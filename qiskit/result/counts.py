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

"""A container class for counts from a circuit execution."""

import re

from qiskit.result import postprocess
from qiskit import exceptions


# NOTE: A dict subclass should not overload any dunder methods like __getitem__
# this can cause unexpected behavior and issues as the cPython dict
# implementation has many standard methods in C for performance and the dunder
# methods are not always used as expected. For example, update() doesn't call
# __setitem__ so overloading __setitem__ would not always provide the expected
# result
class Counts(dict):
    """A class to store a counts result from a circuit execution."""

    def __init__(self, data, time_taken=None, creg_sizes=None,
                 memory_slots=None):
        """Build a counts object

        Args:
            data (dict): The dictionary input for the counts. Where the keys
                represent a measured classical value and the value is an
                integer the number of shots with that result.
                The keys can be one of several formats:

                     * A hexademical string of the form ``"0x4a"``
                     * A bit string prefixed with ``0b`` for example ``'0b1011'``
                     * A bit string formatted across register and memory slots.
                       For example, ``'00 10'``.
                     * A dit string, for example ``'02'``. Note for objects created
                       with dit strings the ``creg_sizes``and ``memory_slots``
                       kwargs don't work and :meth:`hex_outcomes` and
                       :meth:`int_outcomes` also do not work.

            time_taken (float): The duration of the experiment that generated
                the counts
            creg_sizes (list): a nested list where the inner element is a list
                of tuples containing both the classical register name and
                classical register size. For example,
                ``[('c_reg', 2), ('my_creg', 4)]``.
            memory_slots (int): The number of total ``memory_slots`` in the
                experiment.
        Raises:
            TypeError: If the input key type is not an int or string
            QiskitError: If a dit string key is input with creg_sizes and/or
                memory_slots
        """
        bin_data = None
        data = dict(data)
        first_key = next(iter(data.keys()))
        if isinstance(first_key, int):
            self.int_raw = data
            self.hex_raw = {
                hex(key): value for key, value in self.int_raw.items()}
        elif isinstance(first_key, str):
            if first_key.startswith('0x'):
                self.hex_raw = data
                self.int_raw = {
                    int(key, 0): value for key, value in self.hex_raw.items()}
            elif first_key.startswith('0b'):
                self.int_raw = {
                    int(key, 0): value for key, value in data.items()}
                self.hex_raw = {
                    hex(key): value for key, value in self.int_raw.items()}
            else:
                if not creg_sizes and not memory_slots:
                    self.hex_raw = None
                    self.int_raw = None
                    bin_data = data
                else:
                    bitstring_regex = re.compile(r'^[01\s]+$')
                    hex_dict = {}
                    int_dict = {}
                    for bitstring, value in data.items():
                        if not bitstring_regex.search(bitstring):
                            raise exceptions.QiskitError(
                                'Counts objects with dit strings do not '
                                'currently support dit string formatting parameters '
                                'creg_sizes or memory_slots')
                        int_key = int(bitstring.replace(" ", ""), 2)
                        int_dict[int_key] = value
                        hex_dict[hex(int_key)] = value
                    self.hex_raw = hex_dict
                    self.int_raw = int_dict
        else:
            raise TypeError("Invalid input key type %s, must be either an int "
                            "key or string key with hexademical value or bit string")
        header = {}
        self.creg_sizes = creg_sizes
        if self.creg_sizes:
            header['creg_sizes'] = self.creg_sizes
        self.memory_slots = memory_slots
        if self.memory_slots:
            header['memory_slots'] = self.memory_slots
        if not bin_data:
            bin_data = postprocess.format_counts(self.hex_raw, header=header)
        super().__init__(bin_data)
        self.time_taken = time_taken

    def most_frequent(self):
        """Return the most frequent count

        Returns:
            str: The bit string for the most frequent result
        Raises:
            QiskitError: when there is >1 count with the same max counts
        """
        max_value = max(self.values())
        max_values_counts = [x[0] for x in self.items() if x[1] == max_value]
        if len(max_values_counts) != 1:
            raise exceptions.QiskitError(
                "Multiple values have the same maximum counts: %s" %
                ','.join(max_values_counts))
        return max_values_counts[0]

    def hex_outcomes(self):
        """Return a counts dictionary with hexademical string keys

        Returns:
            dict: A dictionary with the keys as hexadecimal strings instead of
                bitstrings
        Raises:
            QiskitError: If the Counts object contains counts for dit strings
        """
        if self.hex_raw:
            return {key.lower(): value for key, value in self.hex_raw.items()}
        else:
            bitstring_regex = re.compile(r'^[01\s]+$')
            out_dict = {}
            for bitstring, value in self.items():
                if not bitstring_regex.search(bitstring):
                    raise exceptions.QiskitError(
                        'Counts objects with dit strings do not '
                        'currently support conversion to hexadecimal')
                int_key = int(bitstring.replace(" ", ""), 2)
                out_dict[hex(int_key)] = value
            return out_dict

    def int_outcomes(self):
        """Build a counts dictionary with integer keys instead of count strings

        Returns:
            dict: A dictionary with the keys as integers instead of bitstrings
        Raises:
            QiskitError: If the Counts object contains counts for dit strings
        """
        if self.int_raw:
            return self.int_raw
        else:
            bitstring_regex = re.compile(r'^[01\s]+$')
            out_dict = {}
            for bitstring, value in self.items():
                if not bitstring_regex.search(bitstring):
                    raise exceptions.QiskitError(
                        'Counts objects with dit strings do not '
                        'currently support conversion to integer')
                int_key = int(bitstring.replace(" ", ""), 2)
                out_dict[int_key] = value
            return out_dict
