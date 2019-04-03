# -*- coding: utf-8 -*-

# Copyright 2019 IBM.
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
"""
Arithmetic Utilities
"""

import numpy as np


def normalize_vector(vector):
    """
    Normalize the input state vector.
    """
    return vector / np.linalg.norm(vector)


def is_power_of_2(num):
    """
    Check if the input number is a power of 2.
    """
    return num != 0 and ((num & (num - 1)) == 0)


def log2(num):
    """
    Compute the log2 of the input number. Use bit operation if the input is a power of 2.
    """
    if is_power_of_2(num):
        ret = 0
        while True:
            if num >> ret == 1:
                return ret
            else:
                ret += 1
    else:
        return np.log2(num)
