# -*- coding: utf-8 -*-
# pylint: disable=redefined-builtin

# Copyright 2018 IBM RESEARCH. All Rights Reserved.
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
"""Helper module for simplified QISKit usage.

The functions in this module provide convenience helpers for accessing commonly
used features of the SDK in a simplified way. They support a small subset of
scenarios and flows: for more advanced usage, it is encouraged to instead
refer to the documentation of each component and use them separately.
"""

from ._wrapper import (available_backends, local_backends, remote_backends,
                       get_backend, compile, execute, register)
