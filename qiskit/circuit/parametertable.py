# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.
"""
Look-up table for varaible parameters in QuantumCircuit.
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

    def __setitem__(self, parameter, value):
        """set the value of parameter

        If the parameter is not already in the table, value should be a list of
        (Instruction, param_index) tuples. __setitem__ will record in the
        ParameterTable that Instruction is dependent on parameter at param_index.

        If the parameter is in the table, value should be either a Number, or a
        { Parameter: value } dict. __setitem__ will update all instructions in
        the ParameterTable with the set value of Parameter.

        Args:
            parameter (Parameter): the parameter to set
            value (Number or dict or list): numeric constant or dictionary of
                (Parameter: value) pairs or list of (Instruction, int) tuples
        """
        if parameter not in self._table:
            for instruction, param_index in value:
                assert isinstance(instruction, Instruction)
                assert isinstance(param_index, int)
            self._table[parameter] = value
        else:
            for (instr, param_index) in self._table[parameter]:
                params = instr.params
                if isinstance(value, dict):
                    for _, this_value in value.items():
                        params[param_index] = this_value
                else:
                    params[param_index] = value
                instr.params = params

    def __delitem__(self, key):
        del self._table[key]

    def __iter__(self):
        return iter(self._table)

    def __len__(self):
        return len(self._table)
