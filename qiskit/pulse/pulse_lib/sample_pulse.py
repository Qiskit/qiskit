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

"""A pulse that is described by complex-valued sample points."""
import warnings
from typing import Callable, Union, List, Optional

import numpy as np

from ..exceptions import PulseError
from .pulse import Pulse


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

    def __eq__(self, other: 'SamplePulse') -> bool:
        """Two SamplePulses are the same if they are of the same type
        and have the same name and samples.

        Args:
            other: Object to compare to.

        Returns:
            True iff self and other are equal.
        """
        return super().__eq__(other) and (self.samples == other.samples).all()

    def __hash__(self):
        return hash(self.samples.tostring())

    def __repr__(self):
        opt = np.get_printoptions()
        np.set_printoptions(threshold=50)
        np.set_printoptions(**opt)
        return '{}({}, name="{}")'.format(self.__class__.__name__, repr(self.samples), self.name)
