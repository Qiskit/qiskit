# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Parametric pulse commands module. These are pulse commands which are described by a specified
parameterization.

If a backend supports parametric pulses, it will have the attribute
`backend.configuration().parametric_pulses`, which is a list of supported pulse shapes, such as
`['gaussian', 'gaussian_square', 'drag']`. A Pulse Schedule, using parametric pulses, which is
assembled for a backend which supports those pulses, will result in a Qobj which is dramatically
smaller than one which uses SamplePulses.

This module can easily be extended to describe more pulse shapes. The new class should:
  - have a descriptive name
  - be a well known and/or well described formula (include the formula in the class docstring)
  - take some parameters (at least `duration`) and validate them, if necessary
  - implement a `get_sample_pulse` method which returns a corresponding SamplePulse in the
    case that it is assembled for a backend which does not support it.

The new pulse must then be registered by the assembler in
`qiskit/qobj/converters/pulse_instruction.py:ParametricPulseShapes`
by following the existing pattern:
    class ParametricPulseShapes(Enum):
        gaussian = commands.Gaussian
        ...
        new_supported_pulse_name = commands.YourPulseCommandClass
"""
import warnings

from qiskit.pulse.pulse_lib import ParametricPulse, Gaussian, GaussianSquare, Drag, ConstantPulse


class ParametricInstruction:
    """Instruction to drive a parametric pulse to an `PulseChannel`."""

    def __init__(self):
        warnings.warn("TODO", DeprecationWarning)
