# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Common utilities for QISKit."""

import logging
import re
import sys
import warnings

def _enable_deprecation_warnings():
    """
    Force the `DeprecationWarning` warnings to be displayed for the qiskit
    module, overriding the system configuration as they are ignored by default
    [1] for end-users.

    TODO: on Python 3.7, this might not be needed due to PEP-0565 [2].

    [1] https://docs.python.org/3/library/warnings.html#default-warning-filters
    [2] https://www.python.org/dev/peps/pep-0565/
    """
    # pylint: disable=invalid-name
    deprecation_filter = ('always', None, DeprecationWarning,
                          re.compile(r'^qiskit\.*', re.UNICODE), 0)

    # Instead of using warnings.simple_filter() directly, the internal
    # _add_filter() function is used for being able to match against the
    # module.
    warnings._add_filter(*deprecation_filter, append=False)


_enable_deprecation_warnings()
