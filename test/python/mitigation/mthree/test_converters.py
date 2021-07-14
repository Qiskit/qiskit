# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test converts"""
from qiskit.test import QiskitTestCase
from qiskit.mitigation.mthree.matrix import bitstring_int
from qiskit.mitigation.mthree.test.converters_testing import (_test_counts_roundtrip,
                                                              _test_counts_to_array)

COUNTS = {'00000': 520,
          '00001': 10,
          '10000': 21,
          '10011': 1,
          '10100': 3,
          '10101': 4,
          '10110': 2,
          '10111': 23,
          '11000': 3,
          '11001': 4,
          '11010': 1,
          '11011': 17,
          '11100': 8,
          '11101': 64,
          '11110': 36,
          '11111': 374,
          '00010': 33,
          '00011': 4,
          '00100': 4,
          '00101': 2,
          '00111': 5,
          '01000': 11,
          '01010': 2,
          '01011': 1,
          '01101': 4,
          '01110': 4,
          '01111': 31}


class TestConverters(QiskitTestCase):
    """Tests conversion routines"""

    def test_counts_converted_properly(self):
        """Tests counts strings are converted properly"""
        # The counts need to be sorted by int value as that is what
        # the cpp_map is doing internally.
        sorted_counts = dict(sorted(COUNTS.items(),
                                    key=lambda item: bitstring_int(item[0])))
        ans_list = list(sorted_counts.keys())
        out_list = _test_counts_to_array(COUNTS)
        self.assertTrue(ans_list == out_list)


    def test_roundtrip_convert(self):
        """Tests converts work roundtrip"""
        shots = sum(COUNTS.values())
        out = _test_counts_roundtrip(COUNTS)
        for key, val in COUNTS.items():
            self.assertTrue(abs(val/shots - out[key]) <= 1e-15)
