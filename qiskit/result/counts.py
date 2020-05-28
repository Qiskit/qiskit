# -*- coding: utf-8 -*-

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

from qiskit.result import postprocess
from qiskit import exceptions


class Counts(dict):
    """A class to store a counts result from a circuit execution."""

    def __init__(self, data, name=None, shots=None, time_taken=None,
                 creg_sizes=None, memory_slots=None, **metadata):
        """Build a counts object

        Args:
            data (dict): The dictionary input for the counts. The key should
                be a hexademical string of the form ``"0x4a"`` representing the
                measured classical value from the experiment and the
                dictionary's value is an integer representing the number of
                shots with that result.
            name (str): A string name for the counts object
            shots (int): The number of shots used in the experiment
            time_taken (float): The duration of the experiment that generated
                the counts
            creg_sizes (list): a nested list where the inner element is a list
                of tuples containing both the classical register name and
                classical register size. For example,
                ``[('c_reg', 2), ('my_creg', 4)]``.
            memory_slots (int): The number of total ``memory_slots`` in the
                experiment.
            metadata: Any arbitrary key value metadata passed in as kwargs.
        """
        self.hex_raw = dict(data)
        header = {}
        self.creg_sizes = creg_sizes
        if self.creg_sizes:
            header['creg_sizes'] = self.creg_sizes
        self.memory_slots = memory_slots
        if self.memory_slots:
            header['memory_slots'] = self.memory_slots
        bin_data = postprocess.format_counts(self.hex_raw, header=header)
        super().__init__(bin_data)
        self.name = name
        self.shots = shots
        self.time_taken = time_taken
        self.metadata = metadata

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

    def int_outcomes(self):
        """Build a counts dictionary with integer keys instead of count strings

        Returns:
            dict: A dictionary with the keys as integers instead of
        """
        return {int(key, 0): value for key, value in self.hex_raw.items()}
