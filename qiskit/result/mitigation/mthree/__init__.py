# This code is part of Mthree.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Mitigation classes
------------------

.. autosummary::
   :toctree: ../stubs/

   M3Mitigation

"""

try:
    from .version import version as __version__
    from .version import openmp
except ImportError:
    __version__ = '0.0.0'
    openmp = False

from .mitigation import M3Mitigation


def about():
    """The M3 version info function.
    """
    print('='*80)
    print('# Matrix-free Measurement Mitigation (M3) version {}'.format(__version__))
    print('# (C) Copyright IBM 2021.')
    print('# Paul Nation, Hwajung Kang, Neereja Sundaresan')
    print('# Jay Gambetta, and Matthew Treinish.')
    print('# Compiled with OpenMP: {}'.format(openmp))
    print('='*80)
