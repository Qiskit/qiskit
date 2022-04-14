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

"""
===========================================
Pulse Library (:mod:`qiskit.pulse.library`)
===========================================

This library provides Pulse users with convenient methods to build Pulse waveforms.

A pulse programmer can choose of one of pulse representation models to create a pulse program.
The :class:`~Waveform` model directly stores waveform data points in a class instance.
This model provides flexibility for the expression of waveform shape and allows us
a rapid prototyping of new control technique. However, this model usually suffers poor memory
efficiency, and it might be hard to scale with large-size quantum processors.
Standard waveform functions are also provided by :mod:`~qiskit.pulse.library.discrete`,
but a user can also directly create a :class:`~Waveform` instance with raw array-like data.

In contrast, the parametric form model, or :class:`~SymbolicPulse`,
only stores a waveform function itself with its parameter values in a class instance,
and thus it provides greater memory efficiency at the price of flexibility of waveforms.
This model also defines a small set of parametric-form pulse subclasses
which is conventionally used in the superconducting quantum processors.
An instance of these subclasses can be serialized in the QPY binary format
while keeping the memory-efficient parametric representation of waveforms.


Pulse Models
============

.. autosummary::
   :toctree: ../stubs/

   Waveform
   SymbolicPulse


Waveform Functions
==================

.. autosummary::
   :toctree: ../stubs/

   constant
   zero
   square
   sawtooth
   triangle
   cos
   sin
   gaussian
   gaussian_deriv
   sech
   sech_deriv
   gaussian_square
   drag


Parametric-form Pulses
======================

.. autosummary::
   :toctree: ../stubs/

   Constant
   Drag
   Gaussian
   GaussianSquare

"""

from .discrete import (
    constant,
    zero,
    square,
    sawtooth,
    triangle,
    cos,
    sin,
    gaussian,
    gaussian_deriv,
    sech,
    sech_deriv,
    gaussian_square,
    drag,
)
from .parametric_pulses import ParametricPulse
from .symbolic_pulses import SymbolicPulse, Gaussian, GaussianSquare, Drag, Constant
from .pulse import Pulse
from .waveform import Waveform
