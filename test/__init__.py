# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""With some utils"""

from ddt import data, unpack
from qiskit.test.utils import generate_cases


def combine(**kwargs):
    """Decorator to create combinations and tests
    @combine(level=[0, 1, 2, 3],
             circuit=[a, b, c, d],
             dsc='Test circuit {circuit.__name__} with level {level}',
             name='{circuit.__name__}_level{level}')
    """

    def deco(func):
        return data(*generate_cases(docstring=func.__doc__, **kwargs))(unpack(func))

    return deco
