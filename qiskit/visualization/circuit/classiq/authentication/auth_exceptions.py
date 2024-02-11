# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Classiq authentication errors"""


class ClassiqAuthenticationError(Exception):
    """Raised when failed to authenticate to Classiq"""

    pass


class ClassiqExpiredTokenError(Exception):
    """Raised when the token used to authenticate to Classiq is expired"""

    pass
