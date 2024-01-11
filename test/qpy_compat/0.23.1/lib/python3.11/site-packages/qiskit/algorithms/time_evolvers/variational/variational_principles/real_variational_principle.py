# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Class for a Real Variational Principle."""

from abc import ABC

from .variational_principle import VariationalPrinciple


class RealVariationalPrinciple(VariationalPrinciple, ABC):
    """Class for a Real Variational Principle. The real variant
    means that we consider real time dynamics."""
