# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
NamedValue, a common interface for components of schedule (Instruction and Schedule) and beyond.
"""
from abc import ABCMeta, abstractmethod
from typing import List
from qiskit.pulse.channels import Channel

# pylint: disable=missing-type-doc


class NamedValue(metaclass=ABCMeta):
    """Common interface for components of schedule and future interconnected classes. """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of NamedValue."""
        pass

    @property
    @abstractmethod
    def channels(self) -> List[Channel]:
        """Return channels used by NamedValue."""
        pass
