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

"""This module contains the SamplePulse and ParametricPulse classes and the Pulse base class.

The SamplePulse is a pulse that is described by complex-valued sample points.

The Parametric Pulses are pulses which are described by a specified
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
from abc import abstractmethod, ABC
from typing import Any, Callable, Union, List, Optional, Dict

import numpy as np

from .channels import PulseChannel
from .exceptions import PulseError


class Pulse(ABC):
    """The abstract superclass for pulses. Pulses are complex-valued waveform envelopes. The
    modulation phase and frequency are specified separately from ``Pulse``s.
    """

    @abstractmethod
    def __init__(self, duration: int, name: Optional[str] = None):
        if not isinstance(duration, (int, np.integer)):
            raise PulseError('Pulse duration should be integer.')
        self.duration = int(duration)
        self.name = name

    @property
    def id(self) -> int:  # pylint: disable=invalid-name
        """Unique identifier for this pulse."""
        return id(self)

    def __call__(self, channel: PulseChannel):
        warnings.warn("Calling `{}` with a channel is deprecated. Instantiate the new `Play` "
                      "instruction directly with a pulse and a channel. In this case, please "
                      "use: `Play({}, {})`.".format(self.__class__.__name__, repr(self), channel),
                      DeprecationWarning)
        from .instructions import Play  # pylint: disable=cyclic-import
        return Play(self, channel)

    @abstractmethod
    def draw(self, dt: float = 1,
             style=None,
             filename: Optional[str] = None,
             interp_method: Optional[Callable] = None,
             scale: float = 1, interactive: bool = False,
             scaling: float = None):
        """Plot the interpolated envelope of pulse.

        Args:
            dt: Time interval of samples.
            style (Optional[PulseStyle]): A style sheet to configure plot appearance
            filename: Name required to save pulse image
            interp_method: A function for interpolation
            scale: Relative visual scaling of waveform amplitudes
            interactive: When set true show the circuit in a new window
                (this depends on the matplotlib backend being used supporting this)
            scaling: Deprecated, see `scale`

        Returns:
            matplotlib.figure: A matplotlib figure object of the pulse envelope
        """
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other: 'Pulse') -> bool:
        return isinstance(other, type(self))

    @abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError


class SamplePulse(Pulse):
    """A pulse specified completely by complex-valued samples; each sample is played for the
    duration of the backend cycle-time, dt.
    """

    def __init__(self, samples: Union[np.ndarray, List[complex]],
                 name: Optional[str] = None,
                 epsilon: float = 1e-7):
        """Create new sample pulse command.

        Args:
            samples: Complex array of the samples in the pulse envelope.
            name: Unique name to identify the pulse.
            epsilon: Pulse sample norm tolerance for clipping.
                If any sample's norm exceeds unity by less than or equal to epsilon
                it will be clipped to unit norm. If the sample
                norm is greater than 1+epsilon an error will be raised.
        """
        samples = np.asarray(samples, dtype=np.complex_)
        self._samples = self._clip(samples, epsilon=epsilon)
        super().__init__(duration=len(samples), name=name)

    @property
    def samples(self) -> np.ndarray:
        """Return sample values."""
        return self._samples

    def _clip(self, samples: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
        """If samples are within epsilon of unit norm, clip sample by reducing norm by (1-epsilon).

        If difference is greater than epsilon error is raised.

        Args:
            samples: Complex array of the samples in the pulse envelope.
            epsilon: Pulse sample norm tolerance for clipping.
                If any sample's norm exceeds unity by less than or equal to epsilon
                it will be clipped to unit norm. If the sample
                norm is greater than 1+epsilon an error will be raised.

        Returns:
            Clipped pulse samples.

        Raises:
            PulseError: If there exists a pulse sample with a norm greater than 1+epsilon.
        """
        samples_norm = np.abs(samples)
        to_clip = (samples_norm > 1.) & (samples_norm <= 1. + epsilon)

        if np.any(to_clip):
            # first try normalizing by the abs value
            clip_where = np.argwhere(to_clip)
            clip_angle = np.angle(samples[clip_where])
            clipped_samples = np.exp(1j*clip_angle, dtype=np.complex_)

            # if norm still exceed one subtract epsilon
            # required for some platforms
            clipped_sample_norms = np.abs(clipped_samples)
            to_clip_epsilon = clipped_sample_norms > 1.
            if np.any(to_clip_epsilon):
                clip_where_epsilon = np.argwhere(to_clip_epsilon)
                clipped_samples_epsilon = np.exp(
                    (1-epsilon)*1j*clip_angle[clip_where_epsilon], dtype=np.complex_)
                clipped_samples[clip_where_epsilon] = clipped_samples_epsilon

            # update samples with clipped values
            samples[clip_where] = clipped_samples
            samples_norm[clip_where] = np.abs(clipped_samples)

        if np.any(samples_norm > 1.):
            raise PulseError('Pulse contains sample with norm greater than 1+epsilon.')

        return samples

    def draw(self, dt: float = 1,
             style=None,
             filename: Optional[str] = None,
             interp_method: Optional[Callable] = None,
             scale: float = 1, interactive: bool = False,
             scaling: float = None):
        """Plot the interpolated envelope of pulse.

        Args:
            dt: Time interval of samples.
            style (Optional[PulseStyle]): A style sheet to configure plot appearance.
            filename: Name required to save pulse image.
            interp_method: A function for interpolation.
            scale: Relative visual scaling of waveform amplitudes.
            interactive: When set true show the circuit in a new window.
                         (This depends on the matplotlib backend being used.)
            scaling: Deprecated, see `scale`,

        Returns:
            matplotlib.figure: A matplotlib figure object of the pulse envelope
        """
        # pylint: disable=invalid-name, cyclic-import
        if scaling is not None:
            warnings.warn(
                'The parameter "scaling" is being replaced by "scale"',
                DeprecationWarning, 3)
            scale = scaling

        from qiskit import visualization

        return visualization.pulse_drawer(self, dt=dt, style=style, filename=filename,
                                          interp_method=interp_method, scale=scale,
                                          interactive=interactive)

    def __eq__(self, other: Pulse) -> bool:
        return super().__eq__(other) and (self.samples == other.samples).all()

    def __hash__(self) -> int:
        return hash(self.samples.tostring())

    def __repr__(self) -> str:
        opt = np.get_printoptions()
        np.set_printoptions(threshold=50)
        np.set_printoptions(**opt)
        return "{}({}{})".format(self.__class__.__name__, repr(self.samples),
                                 ", name='{}'".format(self.name) if self.name is not None else "")

    def __call__(self, channel: PulseChannel):
        warnings.warn("Calling `{}` with a channel is deprecated. Instantiate the new `Play` "
                      "instruction directly with a pulse and a channel. In this case, please "
                      "use: `Play(SamplePulse(samples), {})`."
                      "".format(self.__class__.__name__, channel),
                      DeprecationWarning)
        return super().__call__(channel)


class ParametricPulse(Pulse):
    """The abstract superclass for parametric pulses."""

    @abstractmethod
    def __init__(self, duration: int, name: Optional[str] = None):
        """Create a parametric pulse and validate the input parameters.

        Args:
            duration: Pulse length in terms of the the sampling period `dt`.
            name: Display name for this pulse envelope.
        """
        super().__init__(duration=duration, name=name)
        self.validate_parameters()

    @abstractmethod
    def get_sample_pulse(self) -> SamplePulse:
        """Return a SamplePulse with samples filled according to the formula that the pulse
        represents and the parameter values it contains.
        """
        raise NotImplementedError

    @abstractmethod
    def validate_parameters(self) -> None:
        """
        Validate parameters.

        Raises:
            PulseError: If the parameters passed are not valid.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """Return a dictionary containing the pulse's parameters."""
        pass

    def draw(self, dt: float = 1,
             style=None,
             filename: Optional[str] = None,
             interp_method: Optional[Callable] = None,
             scale: float = 1, interactive: bool = False,
             scaling: float = None):
        """Plot the pulse.

        Args:
            dt: Time interval of samples.
            style (Optional[PulseStyle]): A style sheet to configure plot appearance
            filename: Name required to save pulse image
            interp_method: A function for interpolation
            scale: Relative visual scaling of waveform amplitudes
            interactive: When set true show the circuit in a new window
                (this depends on the matplotlib backend being used supporting this)
            scaling: Deprecated, see `scale`

        Returns:
            matplotlib.figure: A matplotlib figure object of the pulse envelope
        """
        return self.get_sample_pulse().draw(dt=dt, style=style, filename=filename,
                                            interp_method=interp_method, scale=scale,
                                            interactive=interactive)

    def __eq__(self, other: Pulse) -> bool:
        return super().__eq__(other) and self.parameters == other.parameters

    def __hash__(self) -> int:
        return hash(self.parameters[k] for k in sorted(self.parameters))
