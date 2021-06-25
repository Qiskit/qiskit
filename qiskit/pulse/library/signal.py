# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Implements a Signal."""

from typing import Any, Dict, Optional, Union

from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.library.pulse import Pulse
from qiskit.pulse.frame import Frame
from qiskit.pulse.exceptions import PulseError


class Signal:
    """
    A Signal is a Pulse, i.e. a complex-valued waveform envelope, played in a given
    Frame, i.e. a frequency and a phase.
    """

    def __init__(self, pulse: Pulse, frame: Frame, name: Optional[str] = None):
        """
        Args:
            pulse: The envelope of the signal.
            frame: A reference to a frame in which the pulse will be played.
            name: Name of the signal.

        Raises:
            PulseError: if the pulse is not a Pulse.
        """
        if not isinstance(pulse, Pulse):
            raise PulseError("The `pulse` argument to `Signal` must be of type `library.Pulse`.")

        self._pulse = pulse
        self._frame = frame
        self.name = name or f"{pulse.name}_{frame.name}"

    @property
    def pulse(self) -> Pulse:
        """Return the envelope."""
        return self._pulse

    @property
    def frame(self) -> Frame:
        """Return the frame."""
        return self._frame

    @property
    def duration(self) -> Union[int, ParameterExpression]:
        """Return the duration of the signal as the duration of the envelope."""
        return self._pulse.duration

    def is_parameterized(self) -> bool:
        """Determine if there are any parameters in the Signal."""
        param_id = self._frame.identifier[1]
        return self._pulse.is_parameterized() or isinstance(param_id, ParameterExpression)

    @property
    def parameters(self) -> Dict[str, Any]:
        """Return a list of parameters in the Signal."""
        parameters = self._pulse.parameters
        param_id = self._frame.identifier[1]

        if isinstance(param_id, ParameterExpression):
            parameters["index"] = param_id

        return parameters

    def __eq__(self, other: "Signal") -> bool:
        return self._pulse == other._pulse and self.frame == other.frame

    def __repr__(self):
        return f"Signal({self._pulse}, {self._frame})"
