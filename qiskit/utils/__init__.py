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

"""
===============================
Utilities (:mod:`qiskit.utils`)
===============================

.. currentmodule:: qiskit.utils

.. autosummary::
   :toctree: ../stubs/

   deprecate_arguments
   deprecate_function
   local_hardware_info
   is_main_process
   apply_prefix
"""

from .deprecation import _filter_deprecation_warnings
from .deprecation import deprecate_arguments
from .deprecation import deprecate_function
from .multiprocessing import local_hardware_info
from .multiprocessing import is_main_process
from .units import apply_prefix
