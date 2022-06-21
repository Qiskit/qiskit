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

A pulse programmer can choose from one of several :ref:`pulse_models` such as
:class:`~Waveform` and :class:`~SymbolicPulse` to create a pulse program.
The :class:`~Waveform` model directly stores the waveform data points in each class instance.
This model provides the most flexibility to express arbitrary waveforms and allows
a rapid prototyping of new control techniques. However, this model is typically memory
inefficient and might be hard to scale to large-size quantum processors.
Several waveform subclasses are defined by :ref:`waveforms`,
but a user can also directly instantiate the :class:`~Waveform` class with ``samples`` argument
which is usually a complex numpy array or any kind of array-like data.

In contrast, the :class:`~SymbolicPulse` model only stores the function and its parameters
that generate the waveform in a class instance.
It thus provides greater memory efficiency at the price of less flexibility in the waveform.
This model also defines a small set of pulse subclasses in :ref:`symbolic_pulses`
which are commonly used in superconducting quantum processors.
An instance of these subclasses can be serialized in the :ref:`qpy_format`
while keeping the memory-efficient parametric representation of waveforms.
Note that :class:`~Waveform` object can be generated from an instance of
a :class:`~SymbolicPulse` which will set values for the parameters and
sample the parametric expression to create the :class:`~Waveform`.

.. note::

    QPY serialization support for :class:`.SymbolicPulse` is currently not available.
    This feature will be implemented soon in Qiskit terra version 0.21.


.. _pulse_models:

Pulse Models
============

.. autosummary::
   :toctree: ../stubs/

   Waveform
   SymbolicPulse
   ParametricPulse


.. _waveforms:

Waveform Pulse Representation
=============================

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


.. _symbolic_pulses:

Parametric Pulse Representation
===============================

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
