# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Look-up table for variable parameters in QuantumCircuit.
"""
from collections.abc import MutableMapping

from .instruction import Instruction


class ParameterTable(MutableMapping):
    """Class for managing and setting circuit parameters"""

    def __init__(self, *args, **kwargs):
        """
        the structure of _table is,
           {var_object: [(instruction_object, parameter_index), ...]}
        """
        self._table = dict(*args, **kwargs)

    def __getitem__(self, key):
        return self._table[key]

    def __setitem__(self, parameter, instr_params):
        """Sets list of Instructions that depend on Parameter.

        Args:
            parameter (Parameter): the parameter to set
            instr_params (list): List of (Instruction, int) tuples. Int is the
              parameter index at which the parameter appears in the instruction.
        """

        for instruction, param_index in instr_params:
            assert isinstance(instruction, Instruction)
            assert isinstance(param_index, int)
        self._table[parameter] = instr_params

    def __delitem__(self, key):
        del self._table[key]

    def __iter__(self):
        return iter(self._table)

    def __len__(self):
        return len(self._table)

    def __repr__(self):
        return 'ParameterTable({0})'.format(repr(self._table))
