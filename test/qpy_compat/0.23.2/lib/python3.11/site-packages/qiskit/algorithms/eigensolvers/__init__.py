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

"""
=====================================================================
Eigensolvers Package (:mod:`qiskit.algorithms.eigensolvers`)
=====================================================================

.. currentmodule:: qiskit.algorithms.eigensolvers

Eigensolvers
================

.. autosummary::
   :toctree: ../stubs/

   Eigensolver
   NumPyEigensolver
   VQD

Results
=======

 .. autosummary::
    :toctree: ../stubs/

    EigensolverResult
    NumPyEigensolverResult
    VQDResult

"""

from .numpy_eigensolver import NumPyEigensolver, NumPyEigensolverResult
from .eigensolver import Eigensolver, EigensolverResult
from .vqd import VQD, VQDResult

__all__ = [
    "NumPyEigensolver",
    "NumPyEigensolverResult",
    "Eigensolver",
    "EigensolverResult",
    "VQD",
    "VQDResult",
]
