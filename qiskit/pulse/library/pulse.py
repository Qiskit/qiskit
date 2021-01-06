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

"""Pulses are descriptions of waveform envelopes. They can be transmitted by control electronics
to the device.
"""
import warnings

from typing import Dict, Optional, Any, Tuple
from abc import ABC, abstractmethod

import numpy as np

from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse.exceptions import PulseError


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

    @abstractmethod
    def is_parameterized(self) -> bool:
        """Return True iff the instruction is parameterized."""
        raise NotImplementedError

    @abstractmethod
    def assign_parameters(self,
                          value_dict: Dict[ParameterExpression, ParameterValueType]
                          ) -> 'Pulse':
        """Return a new pulse with parameters assigned.

        Args:
            value_dict: A mapping from Parameters to either numeric values or another
                Parameter expression.

        Returns:
            New pulse with updated parameters.
        """
        raise NotImplementedError

    def draw(self,
             style: Optional[Dict[str, Any]] = None,
             backend: Optional['BaseBackend'] = None,
             time_range: Optional[Tuple[int, int]] = None,
             time_unit: str = 'dt',
             show_waveform_info: bool = True,
             plotter: str = 'mpl2d',
             axis: Optional[Any] = None,
             # deprecated arguments, those args were used by legacy drawer.
             dt: Any = None,
             filename: Any = None,
             interp_method: Any = None,
             scale: Any = None,
             interactive: Any = None,
             draw_title: Any = None):
        """Plot the interpolated envelope of pulse.

        Args:
            style: Stylesheet options. This can be dictionary or preset stylesheet classes. See
                :py:class:~`qiskit.visualization.pulse_v2.stylesheets.IQXStandard`,
                :py:class:~`qiskit.visualization.pulse_v2.stylesheets.IQXSimple`, and
                :py:class:~`qiskit.visualization.pulse_v2.stylesheets.IQXDebugging` for details of
                preset stylesheets.
            backend: Backend object to play the input pulse program. If this object is provided,
                the input program is visualized with the details of hardware information.
            time_range: Set horizontal axis limit. Tuple `(tmin, tmax)`.
            time_unit: The unit of specified time range either `dt` or `ns`.
                The unit of `ns` is available only when `backend` object is provided.
            show_waveform_info: A control property to show annotations, i.e. name, of waveforms.
                Set `True` to show additional information about waveforms.
            plotter: Name of plotter API to generate an output image.
                One of following APIs should be specified::

                    mpl2d: Matplotlib API for 2D image generation.
                        Matplotlib API to generate 2D image. Charts are placed along y axis with
                        vertical offset. This API takes matplotlib.axes.Axes as `axis` input.

                `axis` and `style` kwargs may depend on the plotter.
            axis: Arbitrary object passed to the plotter. If this object is provided,
                the plotters uses given `axis` instead of internally initializing a figure object.
                This object format depends on the plotter. See plotters section for details.

            dt: Deprecated. This argument is used by the legacy pulse drawer.
            filename: Deprecated. This argument is used by the legacy pulse drawer.
                To save output image, you can call `.savefig` method with
                returned Matplotlib Figure object.
            interp_method: Deprecated. This argument is used by the legacy pulse drawer.
            scale: Deprecated. This argument is used by the legacy pulse drawer.
            interactive: Deprecated. This argument is used by the legacy pulse drawer.
            draw_title: Deprecated. This argument is used by the legacy pulse drawer.

        Returns:
            matplotlib.figure: A matplotlib figure object of the pulse envelope
        """
        # pylint: disable=invalid-name, cyclic-import, missing-return-type-doc
        from qiskit.visualization import pulse_drawer_v2

        legacy_args = (dt, filename, interp_method, scale, interactive, draw_title)

        if any(arg is not None for arg in legacy_args):
            warnings.warn('Legacy pulse drawer is deprecated with some arguments. '
                          'Please check the API document of new pulse drawer '
                          '`qiskit.visualization.pulse_drawer_v2`.',
                          DeprecationWarning)

        if filename:
            warnings.warn('File saving is delegated to the plotter software in new drawer. '
                          'If you specify matplotlib plotter family to `plotter` argument, '
                          'you can call `savefig` method with the returned Figure object.',
                          DeprecationWarning)

        return pulse_drawer_v2(program=self,
                               style=style,
                               backend=backend,
                               time_range=time_range,
                               time_unit=time_unit,
                               show_waveform_info=show_waveform_info,
                               plotter=plotter,
                               axis=axis)

    @abstractmethod
    def __eq__(self, other: 'Pulse') -> bool:
        return isinstance(other, type(self))

    @abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError
