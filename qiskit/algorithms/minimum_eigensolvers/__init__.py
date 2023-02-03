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
============================================================================
Minimum Eigensolvers Package (:mod:`qiskit.algorithms.minimum_eigensolvers`)
============================================================================

.. currentmodule:: qiskit.algorithms.minimum_eigensolvers

Minimum Eigensolvers
====================
.. autosummary::
   :toctree: ../stubs/

   MinimumEigensolver
   NumPyMinimumEigensolver
   VQE
   AdaptVQE
   SamplingMinimumEigensolver
   SamplingVQE
   QAOA

.. autosummary::
   :toctree: ../stubs/

   MinimumEigensolverResult
   NumPyMinimumEigensolverResult
   VQEResult
   AdaptVQEResult
   SamplingMinimumEigensolverResult
   SamplingVQEResult
"""

from .adapt_vqe import AdaptVQE, AdaptVQEResult
from .minimum_eigensolver import MinimumEigensolver, MinimumEigensolverResult
from .numpy_minimum_eigensolver import NumPyMinimumEigensolver, NumPyMinimumEigensolverResult
from .vqe import VQE, VQEResult
from .sampling_mes import SamplingMinimumEigensolver, SamplingMinimumEigensolverResult
from .sampling_vqe import SamplingVQE, SamplingVQEResult
from .qaoa import QAOA

__all__ = [
    "AdaptVQE",
    "AdaptVQEResult",
    "MinimumEigensolver",
    "MinimumEigensolverResult",
    "NumPyMinimumEigensolver",
    "NumPyMinimumEigensolverResult",
    "VQE",
    "VQEResult",
    "SamplingMinimumEigensolver",
    "SamplingMinimumEigensolverResult",
    "SamplingVQE",
    "SamplingVQEResult",
    "QAOA",
]
