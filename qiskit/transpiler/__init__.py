# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Utils for transpiler."""
import os
from ._passmanager import PassManager, FlowController
from ._propertyset import PropertySet
from ._transpilererror import TranspilerError, TranspilerAccessError
from ._fencedobjs import FencedDAGCircuit, FencedPropertySet
from ._basepasses import AnalysisPass, TransformationPass
from ._transpiler import transpile, transpile_dag
from ._parallel import parallel_map

# Set parallel environmental variable
os.environ['QISKIT_IN_PARALLEL'] = 'FALSE'
