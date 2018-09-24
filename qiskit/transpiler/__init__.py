# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Utils for transpiler."""
import os
from ._passmanager import PassManager
from ._transpilererror import TranspilerError

# pylint: disable=redefined-builtin
from ._transpiler import compile, transpile

from ._parallel import parallel_map
from ._progressbar import TextProgressBar

# Set parallel ennvironmental variable
os.environ['QISKIT_IN_PARALLEL'] = 'FALSE'
