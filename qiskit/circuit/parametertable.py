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
from functools import cmp_to_key

from .parameterexpression import ParameterExpression
from .exceptions import CircuitError
from .instruction import Instruction
from .parametervector import ParameterVectorElement


class ParameterTable(dict):
    """Class for managing and setting circuit parameters"""

    __slots__ = ['_names']

    def __init__(self, *table):
        """
        the structure of table is,
           {var_object: [(instruction_object, parameter_index)]}
        """
        super().__init__(*table)
        self._names = {param.name for param in self.keys()}

    def __repr__(self):
        return 'ParameterTable({})'.format(super().__repr__())

    def clear(self) -> None:
        super().clear()
        self._names.clear()

    def parameters(self):
        """Return parameters contained in ParameterTable"""
        return Parameters(self.keys())

    # todo deprecate?
    def get_keys(self):
        """Return parameters contained in ParameterTable"""
        return self.parameters()

    def update(self, instruction: Instruction) -> Instruction:
        """Update ParameterTable with instruction."""
        for i, param in enumerate(instruction.params):
            if isinstance(param, ParameterExpression):
                for parameter in param.parameters:
                    if parameter in self:
                        if not self._has_instruction_spec(parameter, instruction, i):
                            self[parameter].append((instruction, i))
                    else:
                        if parameter.name in self._names:
                            raise CircuitError(
                                'Name conflict on adding parameter: {}'.format(parameter.name))
                        self[parameter] = [(instruction, i)]
                    self._names.add(parameter.name)
        return instruction

    def _has_instruction_spec(self, parameter, instruction, index):
        for instr, i in self[parameter]:
            if instr is instruction and i == index:
                return True
        return False


class Parameters(set):
    """Class to provide ordered set of parameters.
    Derives from a set but implements methods to provide iteration ordered by name.
    """

    __slots__ = ['_sorted_list', '_updated']

    def __init__(self, *s):
        super().__init__(*s)
        self._sorted_list = []
        self._updated = True

    def __repr__(self):
        """Format the class as string."""
        self._sort()
        return f'Parameters({self._sorted_list})'

    def __iter__(self):
        self._sort()
        return iter(self._sorted_list)

    def __getitem__(self, item):
        self._sort()
        return self._sorted_list[item]

    def _sort(self):
        if self._updated:
            self._sorted_list = sorted(super().__iter__(), key=cmp_to_key(self._compare_parameters))
            self._updated = False

    def union(self, *s) -> 'Parameters':
        return Parameters(super().union(*s))

    def update(self, *s):
        super().update(*s)
        self._updated = True

    def intersection_update(self, *s):
        super().intersection_update(*s)
        self._updated = True

    def difference_update(self, *s):
        super().difference_update(*s)
        self._updated = True

    def symmetric_difference_update(self, *s):
        super().symmetric_difference_update(*s)
        self._updated = True

    def add(self, elem):
        super().add(elem)
        self._updated = True

    def remove(self, elem):
        super().remove(elem)
        self._updated = True

    def discard(self, elem):
        super().discard(elem)
        self._updated = True

    def clear(self):
        super().clear()
        self._updated = True

    def pop(self, index=-1):
        """ Remove and return parameter at index (default last). """
        self._sort()
        elem = self._sorted_list.pop(index)
        super().discard(elem)
        self._updated = True
        return elem

    @staticmethod
    def _standard_compare(value1, value2):
        if value1 < value2:
            return -1
        if value1 > value2:
            return 1
        return 0

    def _compare_parameters(self, parm1, parm2):
        if isinstance(parm1, ParameterVectorElement) and isinstance(parm2, ParameterVectorElement):
            # if they belong to a vector with the same name, sort by index
            if parm1.vector.name == parm2.vector.name:
                return self._standard_compare(parm1.index, parm2.index)
        # else sort by name
        return self._standard_compare(parm1.name, parm2.name)
