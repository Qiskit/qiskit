# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=redefined-builtin

"""Helper module for simplified QISKit usage.

The functions in this module provide convenience helpers for accessing commonly
used features of the SDK in a simplified way. They support a small subset of
scenarios and flows: for more advanced usage, it is encouraged to instead
refer to the documentation of each component and use them separately.
"""

from ._wrapper import (compile, execute, load_qasm_string, load_qasm_file, qobj_to_circuits)

# To be deprecated methods
from ._wrapper import (available_backends, get_backend, register, unregister, least_busy,
                       registered_providers)
