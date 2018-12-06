# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Module for parallel functionality.
"""

import os

# Set parallel environmental variable
os.environ['QISKIT_IN_PARALLEL'] = 'FALSE'
