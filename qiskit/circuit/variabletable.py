# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.
"""
Look-up table for varaible parameters in QuantumCircuit to support fast
"""
from collections import MutableMapping


class VariableTable(MutableMapping):
    """Class for managing and setting circuit variables"""

    def __init__(self, *args, **kwargs):
        """
        the structure of _table is,
           {var_object: [(instruction_object, parameter_index), ...]}
        """
        self._table = dict(*args, **kwargs)

    def __getitem__(self, key):
        return self._table[key]

    def __setitem__(self, key, value):
        """set the value of variable

        Args:
            key (Object): the variable to set
            value (Number or dict): numeric constant or dictionary of
                (Symbol:value) pairs
        """
        if key not in self._table:
            self._table[key] = value
        else:
            for (instr, param_index) in self._table[key]:
                params = instr.params
                if isinstance(value, dict):
                    params[param_index] = params[param_index].evalf(subs=value)
                else:
                    params[param_index] = value
                instr.params = params

    def __delitem__(self, key):
        del self._table[key]

    def __iter__(self):
        return iter(self._table)

    def __len__(self):
        return len(self._table)
