# -*- coding: utf-8 -*-

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

from abc import ABC
from abc import abstractmethod


class Properties(ABC):
    """Base properties object

    This class is the abstract class that backend properties are based on.
    If a backend has properties defined these are static attributes that
    define the characteristics of a backend. They are not a requried field
    but if it is defined the intent is that they are static characteristics
    of the backend for the lifetime of the backend object.
    """

    @property
    @abstractmethod
    def backend_version(self):
        pass
