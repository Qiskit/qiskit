# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Exception for errors raised by the transpiler.
"""
from qiskit import QISKitError


class TranspilerError(QISKitError):
    """Exceptions raised during transpilation"""
