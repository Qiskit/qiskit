# This code is part of Qiskit.
#
# (C) Copyright IBM 2017 - 2023.
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

.. currentmodule:: qiskit.synthesis

.. _evolution_synthesis:

Evolution Synthesis
===================

.. autosummary::
   :toctree: ../stubs/

   EvolutionSynthesis
   ProductFormula
   LieTrotter
   SuzukiTrotter
   MatrixExponential
   QDrift

Linear Function Synthesis
=========================

.. autofunction:: synth_cnot_count_full_pmh
.. autofunction:: synth_cnot_depth_line_kms

Linear-Phase Synthesis
======================

.. autofunction:: synth_cz_depth_line_mr
.. autofunction:: synth_cx_cz_depth_line_my

Permutation Synthesis
=====================

.. autofunction:: synth_permutation_depth_lnn_kms
.. autofunction:: synth_permutation_basic
.. autofunction:: synth_permutation_acg

Clifford Synthesis
==================

.. autofunction:: synth_clifford_full
.. autofunction:: synth_clifford_ag
.. autofunction:: synth_clifford_bm
.. autofunction:: synth_clifford_greedy
.. autofunction:: synth_clifford_layers
.. autofunction:: synth_clifford_depth_lnn

CNOTDihedral Synthesis
======================

.. autofunction:: synth_cnotdihedral_full
.. autofunction:: synth_cnotdihedral_two_qubits
.. autofunction:: synth_cnotdihedral_general

Stabilizer State Synthesis
==========================

.. autofunction:: synth_stabilizer_layers
.. autofunction:: synth_stabilizer_depth_lnn

Discrete Basis Synthesis
========================

.. autosummary::
   :toctree: ../stubs/

   SolovayKitaevDecomposition

.. autofunction:: generate_basic_approximations

"""

from .evolution import (
    EvolutionSynthesis,
    ProductFormula,
    LieTrotter,
    SuzukiTrotter,
    MatrixExponential,
    QDrift,
)

from .permutation import (
    synth_permutation_depth_lnn_kms,
    synth_permutation_basic,
    synth_permutation_acg,
)
from .linear import (
    synth_cnot_count_full_pmh,
    synth_cnot_depth_line_kms,
)
from .linear_phase import synth_cz_depth_line_mr, synth_cx_cz_depth_line_my, synth_cnot_phase_aam
from .clifford import (
    synth_clifford_full,
    synth_clifford_ag,
    synth_clifford_bm,
    synth_clifford_greedy,
    synth_clifford_layers,
    synth_clifford_depth_lnn,
)
from .cnotdihedral import (
    synth_cnotdihedral_full,
    synth_cnotdihedral_two_qubits,
    synth_cnotdihedral_general,
)
from .stabilizer import synth_stabilizer_layers, synth_stabilizer_depth_lnn
from .discrete_basis import SolovayKitaevDecomposition, generate_basic_approximations
