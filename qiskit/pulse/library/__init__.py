# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

r"""
=====================================================
Pulse Library (waveforms :mod:`qiskit.pulse.library`)
=====================================================

This library provides Pulse users with convenient methods to build Pulse waveforms.

Arbitrary waveforms can be described with :py:class:`~qiskit.pulse.library.Waveform`\ s.

The :py:mod:`~qiskit.pulse.library.discrete` module will generate
:py:class:`~qiskit.pulse.library.Waveform`\ s for common waveform envelopes.

The symbolic pulses, :py:class:`~qiskit.pulse.library.Gaussian`,
:py:class:`~qiskit.pulse.library.GaussianSquare`, :py:class:`~qiskit.pulse.library.Drag` and
:py:class:`~qiskit.pulse.library.Constant` will generate symbolic descriptions of
those pulses, which can greatly reduce the size of the job sent to the backend and
ensure serialization of these pulse data in QPY binary format.

.. autosummary::
   :toctree: ../stubs/

   ~qiskit.pulse.library.discrete
   Waveform
   Constant
   Drag
   Gaussian
   GaussianSquare

"""
from .discrete import *
from .parametric_pulses import ParametricPulse
from .symbolic_pulses import SymbolicPulse, Gaussian, GaussianSquare, Drag, Constant
from .pulse import Pulse
from .waveform import Waveform
