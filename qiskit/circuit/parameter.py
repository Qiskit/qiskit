# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.
"""
Parameter Class for variable parameters.
"""


class Parameter():
    """Parameter Class for variable parameters"""
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        """Returns the name of the Parameter."""
        return self._name

    def __str__(self):
        return self.name

    def __copy__(self):
        return self

    def __deepcopy__(self, memo=None):
        return self

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.name)
