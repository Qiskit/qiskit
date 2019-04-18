# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import warnings

from qiskit.util import *

warnings.warn('The qiskit._util module is deprecated and has been renamed '
              'qiskit.util. Please update your imports as qiskit._util will be'
              'removed in Qiskit Terra 0.9.', DeprecationWarning)
