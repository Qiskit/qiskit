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

"""
This module implements the abstract base class for algorithm results.
"""

from typing import Optional, Dict, Tuple
from abc import ABC
import collections


class AlgorithmResult(ABC, collections.UserDict):
    """ Abstract Base Class for algorithm results."""

    def __init__(self, a_dict: Optional[Dict] = None) -> None:
        super().__init__()
        if a_dict:
            self.data.update(a_dict)

    def __setitem__(self, key: object, item: object) -> None:
        raise TypeError("'__setitem__' invalid for this object.")

    def __delitem__(self, key: object) -> None:
        raise TypeError("'__delitem__' invalid for this object.")

    def clear(self) -> None:
        raise TypeError("'clear' invalid for this object.")

    def pop(self, key: object, default: Optional[object] = None) -> object:
        raise TypeError("'pop' invalid for this object.")

    def popitem(self) -> Tuple[object, object]:
        raise TypeError("'popitem' invalid for this object.")

    def update(self, *args, **kwargs) -> None:
        raise TypeError("'update' invalid for this object.")

    def __contains__(self, key: object) -> bool:
        # subclasses have special __getitem__
        try:
            _ = self.__getitem__(key)
            return True
        except KeyError:
            return False
