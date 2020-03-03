# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Sample pulse.
"""
from typing import Callable, Union, List, Optional

import warnings
import numpy as np

from qiskit.pulse.channels import PulseChannel
from qiskit.pulse.exceptions import PulseError

from .instruction import Instruction
from .pulse_command import PulseCommand


class SamplePulse(PulseCommand):
    """Container for functional pulse."""

    def __init__(self, samples: Union[np.ndarray, List[complex]], name: Optional[str] = None,
                 epsilon: float = 1e-7):
        """Create new sample pulse command.

        Args:
            samples: Complex array of pulse envelope
            name: Unique name to identify the pulse
            epsilon: Pulse sample norm tolerance for clipping.
                If any sample's norm exceeds unity by less than or equal to epsilon
                it will be clipped to unit norm. If the sample
                norm is greater than 1+epsilon an error will be raised
        """
        super().__init__(duration=len(samples))

        samples = np.asarray(samples, dtype=np.complex_)

        self._samples = self._clip(samples, epsilon=epsilon)
        self._name = SamplePulse.create_name(name)

    @property
    def samples(self):
        """Return sample values."""
        return self._samples

    def _clip(self, samples: np.ndarray, epsilon: float = 1e-7):
        """If samples are within epsilon of unit norm, clip sample by reducing norm by (1-epsilon).

        If difference is greater than epsilon error is raised.

        Args:
            samples: Complex array of pulse envelope
            epsilon: Pulse sample norm tolerance for clipping.
                If any sample's norm exceeds unity by less than or equal to epsilon
                it will be clipped to unit norm. If the sample
                norm is greater than 1+epsilon an error will be raised

        Returns:
            np.ndarray: Clipped pulse samples
        Raises:
            PulseError: If there exists a pulse sample with a norm greater than 1+epsilon
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
             style: Optional['PulseStyle'] = None,
             filename: Optional[str] = None,
             interp_method: Optional[Callable] = None,
             scale: float = 1, interactive: bool = False,
             scaling: float = None):
        """Plot the interpolated envelope of pulse.

        Args:
            dt: Time interval of samples.
            style: A style sheet to configure plot appearance
            filename: Name required to save pulse image
            interp_method: A function for interpolation
            scale: Relative visual scaling of waveform amplitudes
            interactive: When set true show the circuit in a new window
                (this depends on the matplotlib backend being used supporting this)
            scaling: Deprecated, see `scale`

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

    def __eq__(self, other: 'SamplePulse'):
        """Two SamplePulses are the same if they are of the same type
        and have the same name and samples.

        Args:
            other: other SamplePulse

        Returns:
            bool: are self and other equal
        """
        return super().__eq__(other) and (self.samples == other.samples).all()

    def __hash__(self):
        return hash((super().__hash__(), self.samples.tostring()))

    def __repr__(self):
        opt = np.get_printoptions()
        np.set_printoptions(threshold=50)
        repr_str = '%s(samples=%s, name="%s")' % (self.__class__.__name__,
                                                  repr(self.samples),
                                                  self.name)
        np.set_printoptions(**opt)
        return repr_str

    # pylint: disable=arguments-differ
    def to_instruction(self, channel: PulseChannel,
                       name: Optional[str] = None) -> 'PulseInstruction':
        return PulseInstruction(self, channel, name=name)
    # pylint: enable=arguments-differ


class PulseInstruction(Instruction):
    """Instruction to drive a pulse to an `PulseChannel`."""

    def __init__(self, command: SamplePulse, channel: PulseChannel, name: Optional[str] = None):
        super().__init__(command, channel, name=name)
