# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This code was originally copied from the qiskit-ignis repsoitory see:
# https://github.com/Qiskit/qiskit-ignis/blob/b91066c72171bcd55a70e6e8993b813ec763cf41/qiskit/ignis/mitigation/measurement/__init__.py
# it was migrated as qiskit-ignis is being deprecated

"""
=============================================================
Measurement Mitigation Utils (:mod:`qiskit.utils.mitigation`)
=============================================================

.. currentmodule:: qiskit.utils.mitigation

.. deprecated:: 0.24.0
    This module is deprecated and will be removed no sooner than 3 months
    after the release date. For code migration guidelines,
    visit https://qisk.it/qi_migration.

.. warning::

    The user-facing API stability of this module is not guaranteed except for
    its use with the :class:`~qiskit.utils.QuantumInstance` (i.e. using the
    :class:`~qiskit.utils.mitigation.CompleteMeasFitter` or
    :class:`~qiskit.utils.mitigation.TensoredMeasFitter` classes as values for the
    ``meas_error_mitigation_cls``). The rest of this module should be treated as
    an internal private API that can not be relied upon.

Measurement correction
======================

The measurement calibration is used to mitigate measurement errors.
The main idea is to prepare all :math:`2^n` basis input states and compute
the probability of measuring counts in the other basis states.
From these calibrations, it is possible to correct the average results
of another experiment of interest. These tools are intended for use solely
with the :class:`~qiskit.utils.QuantumInstance` class as part of
:mod:`qiskit.algorithms` and :mod:`qiskit.opflow`.

.. autosummary::
   :toctree: ../stubs/

   CompleteMeasFitter
   TensoredMeasFitter
"""

# Measurement correction functions
from .circuits import complete_meas_cal, tensored_meas_cal
from .fitters import CompleteMeasFitter, TensoredMeasFitter
