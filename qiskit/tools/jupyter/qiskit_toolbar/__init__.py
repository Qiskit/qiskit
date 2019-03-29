# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Qiskit Jupyter toolbar extension"""

def _jupyter_nbextension_paths():
    return [dict(section="notebook",
                 # relative path to the module directory
                 src="",
                 # directory in the `nbextension/` namespace
                 dest="qiskit_toolbar",
                 require='qiskit_toolbar/main')]
