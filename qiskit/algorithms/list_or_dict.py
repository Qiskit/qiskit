# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Introduced new type to maintain readability."""

from typing import TypeVar, List, Union, Optional, Dict

_T = TypeVar("_T")  # Pylint does not allow single character class names.
ListOrDict = Union[List[Optional[_T]], Dict[str, _T]]
