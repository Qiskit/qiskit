# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
===========================================
Circuit Synthesis (:mod:`qiskit.synthesis`)
===========================================

.. automodule:: qiskit.synthesis.discrete_basis

.. currentmodule:: qiskit.synthesis

Evolution Synthesis
===================

.. autosummary::
   :toctree: ../stubs/

   EvolutionSynthesis
   ProductFormula
   LieTrotter
   SuzukiTrotter
   MatrixExponential

Linear Function Synthesis
=========================
.. autosummary::
   :toctree: ../stubs/

    synth_cnot_count_full_pmh
    synth_cnot_depth_line_kms

Permutation Synthesis
=====================

.. autosummary::
   :toctree: ../stubs/

   synth_permutation_depth_lnn_kms

Clifford Synthesis
==================

.. autosummary::
   :toctree: ../stubs/

   synth_clifford_full
   synth_clifford_ag
   synth_clifford_bm
   synth_clifford_greedy

CNOTDihedral Synthesis
=======================

.. autosummary::
   :toctree: ../stubs/

   synth_cnotdihedral_full
   synth_cnotdihedral_two_qubits
   synth_cnotdihedral_general

"""

from .evolution import (
    EvolutionSynthesis,
    ProductFormula,
    LieTrotter,
    SuzukiTrotter,
    MatrixExponential,
    QDrift,
)

from .linear import synth_cnot_count_full_pmh, synth_cnot_depth_line_kms
from .permutation import synth_permutation_depth_lnn_kms
from .clifford import (
    synth_clifford_full,
    synth_clifford_ag,
    synth_clifford_bm,
    synth_clifford_greedy,
)
from .cnotdihedral import (
    synth_cnotdihedral_full,
    synth_cnotdihedral_two_qubits,
    synth_cnotdihedral_general,
)
