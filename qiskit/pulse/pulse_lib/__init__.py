# -*- coding: utf-8 -*-

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

"""This library provides Pulse users with convenient methods to build Pulse waveforms.

Arbitrary waveforms can be described with :py:class:`~qiskit.pulse.pulse_lib.SamplePulse` s.

The :py:mod:`~qiskit.pulse.pulse_lib.discrete` module will generate
:py:class:`~qiskit.pulse.pulse_lib.SamplePulse` s for common waveform envelopes.

The parametric pulses, :py:class:`~qiskit.pulse.pulse_lib.Gaussian`,
:py:class:`~qiskit.pulse.pulse_lib.GaussianSquare`, :py:class:`~qiskit.pulse.pulse_lib.Drag` and
:py:class:`~qiskit.pulse.pulse_lib.Constant` will generate parameterized descriptions of
those pulses, which can greatly reduce the size of the job sent to the backend.

.. autosummary::
   :toctree: ../stubs/

   ~qiskit.pulse.pulse_lib.discrete
   SamplePulse
   Constant
   Drag
   Gaussian
   GaussianSquare

"""

from .discrete import *
from .parametric_pulses import (ParametricPulse, Gaussian, GaussianSquare,
                                Drag, Constant, ConstantPulse)
from .pulse import Pulse
from .sample_pulse import SamplePulse
