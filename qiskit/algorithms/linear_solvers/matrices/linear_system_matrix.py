# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An abstract class for linear systems solvers in Qiskit's aqua module."""
from abc import ABC, abstractmethod
from typing import Union, Optional, List
import numpy as np


class LinearSystemMatrix(ABC):
    """An abstract class for linear system matrices in Qiskit.(Placeholder)"""
    def __init__(self):
        pass

    @abstractmethod
    def power(self, power: int):
        raise NotImplementedError