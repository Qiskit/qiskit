# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=no-member,invalid-name,redefined-outer-name,missing-docstring

"""
A two-ways dict to represent a layout.

DEPRECATED IN TERRA 0.8+
"""
import warnings


def Layout(input_dict=None):
    warnings.warn('qiskit.mapper.Layout has moved to '
                  'qiskit.transpiler.Layout.', DeprecationWarning)
    from qiskit.transpiler.layout import Layout
    return Layout(input_dict)
