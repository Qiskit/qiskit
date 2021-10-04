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

# pylint: disable=missing-return-doc

"""Module for builtin discrete pulses.

Note the sampling strategy use for all discrete pulses is ``midpoint``.
"""
from typing import Optional

from ..exceptions import PulseError
from .waveform import Waveform
from . import continuous
from . import samplers


_sampled_constant_pulse = samplers.midpoint(continuous.constant)


def constant(duration: int, amp: complex, name: Optional[str] = None) -> Waveform:
    r"""Generates constant-sampled :class:`~qiskit.pulse.Waveform`.

    For :math:`A=` ``amp``, samples from the function:

    .. math::

        f(x) = A

    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Complex pulse amplitude.
        name: Name of pulse.
    """
    return _sampled_constant_pulse(duration, amp, name=name)


_sampled_zero_pulse = samplers.midpoint(continuous.zero)


def zero(duration: int, name: Optional[str] = None) -> Waveform:
    """Generates zero-sampled :class:`~qiskit.pulse.Waveform`.

    Samples from the function:

    .. math::

        f(x) = 0

    Args:
        duration: Duration of pulse. Must be greater than zero.
        name: Name of pulse.
    """
    return _sampled_zero_pulse(duration, name=name)


_sampled_square_pulse = samplers.midpoint(continuous.square)


def square(
    duration: int, amp: complex, freq: float = None, phase: float = 0, name: Optional[str] = None
) -> Waveform:
    r"""Generates square wave :class:`~qiskit.pulse.Waveform`.

    For :math:`A=` ``amp``, :math:`T=` ``period``, and :math:`\phi=` ``phase``,
    applies the `midpoint` sampling strategy to generate a discrete pulse sampled from
    the continuous function:

    .. math::

        f(x) = A \text{sign}\left[ \sin\left(\frac{2 \pi x}{T} + 2\phi\right) \right]

    with the convention :math:`\text{sign}(0) = 1`.


    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Pulse amplitude. Wave range is :math:`[-` ``amp`` :math:`,` ``amp`` :math:`]`.
        freq: Pulse frequency, units of 1./dt. If ``None`` defaults to 1./duration.
        phase: Pulse phase.
        name: Name of pulse.
    """
    if freq is None:
        freq = 1.0 / duration

    return _sampled_square_pulse(duration, amp, freq, phase=phase, name=name)


_sampled_sawtooth_pulse = samplers.midpoint(continuous.sawtooth)


def sawtooth(
    duration: int, amp: complex, freq: float = None, phase: float = 0, name: Optional[str] = None
) -> Waveform:
    r"""Generates sawtooth wave :class:`~qiskit.pulse.Waveform`.

    For :math:`A=` ``amp``, :math:`T=` ``period``, and :math:`\phi=` ``phase``,
    applies the `midpoint` sampling strategy to generate a discrete pulse sampled from
    the continuous function:

    .. math::

        f(x) = 2 A \left( g(x) - \left\lfloor \frac{1}{2} + g(x) \right\rfloor\right)

    where :math:`g(x) = x/T + \phi/\pi`.

    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Pulse amplitude. Wave range is :math:`[-` ``amp`` :math:`,` ``amp`` :math:`]`.
        freq: Pulse frequency, units of 1./dt. If ``None`` defaults to 1./duration.
        phase: Pulse phase.
        name: Name of pulse.

    Example:
        .. jupyter-execute::

            import matplotlib.pyplot as plt
            from qiskit.pulse.library import sawtooth
            import numpy as np

            duration = 100
            amp = 1
            freq = 1 / duration
            sawtooth_wave = np.real(sawtooth(duration, amp, freq).samples)
            plt.plot(range(duration), sawtooth_wave)
    """
    if freq is None:
        freq = 1.0 / duration

    return _sampled_sawtooth_pulse(duration, amp, freq, phase=phase, name=name)


_sampled_triangle_pulse = samplers.midpoint(continuous.triangle)


def triangle(
    duration: int, amp: complex, freq: float = None, phase: float = 0, name: Optional[str] = None
) -> Waveform:
    r"""Generates triangle wave :class:`~qiskit.pulse.Waveform`.

    For :math:`A=` ``amp``, :math:`T=` ``period``, and :math:`\phi=` ``phase``,
    applies the `midpoint` sampling strategy to generate a discrete pulse sampled from
    the continuous function:

    .. math::

        f(x) = A \left(-2\left|\text{sawtooth}(x, A, T, \phi)\right| + 1\right)

    This a non-sinusoidal wave with linear ramping.

    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Pulse amplitude. Wave range is :math:`[-` ``amp`` :math:`,` ``amp`` :math:`]`.
        freq: Pulse frequency, units of 1./dt. If ``None`` defaults to 1./duration.
        phase: Pulse phase.
        name: Name of pulse.

    Example:
        .. jupyter-execute::

            import matplotlib.pyplot as plt
            from qiskit.pulse.library import triangle
            import numpy as np

            duration = 100
            amp = 1
            freq = 1 / duration
            triangle_wave = np.real(triangle(duration, amp, freq).samples)
            plt.plot(range(duration), triangle_wave)
    """
    if freq is None:
        freq = 1.0 / duration

    return _sampled_triangle_pulse(duration, amp, freq, phase=phase, name=name)


_sampled_cos_pulse = samplers.midpoint(continuous.cos)


def cos(
    duration: int, amp: complex, freq: float = None, phase: float = 0, name: Optional[str] = None
) -> Waveform:
    r"""Generates cosine wave :class:`~qiskit.pulse.Waveform`.

    For :math:`A=` ``amp``, :math:`\omega=` ``freq``, and :math:`\phi=` ``phase``,
    applies the `midpoint` sampling strategy to generate a discrete pulse sampled from
    the continuous function:

    .. math::

        f(x) = A \cos(2 \pi \omega x + \phi)

    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Pulse amplitude.
        freq: Pulse frequency, units of 1/dt. If ``None`` defaults to single cycle.
        phase: Pulse phase.
        name: Name of pulse.
    """
    if freq is None:
        freq = 1 / duration

    return _sampled_cos_pulse(duration, amp, freq, phase=phase, name=name)


_sampled_sin_pulse = samplers.midpoint(continuous.sin)


def sin(
    duration: int, amp: complex, freq: float = None, phase: float = 0, name: Optional[str] = None
) -> Waveform:
    r"""Generates sine wave :class:`~qiskit.pulse.Waveform`.

    For :math:`A=` ``amp``, :math:`\omega=` ``freq``, and :math:`\phi=` ``phase``,
    applies the `midpoint` sampling strategy to generate a discrete pulse sampled from
    the continuous function:

    .. math::

        f(x) = A \sin(2 \pi \omega x + \phi)

    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Pulse amplitude.
        freq: Pulse frequency, units of 1/dt. If ``None`` defaults to single cycle.
        phase: Pulse phase.
        name: Name of pulse.
    """
    if freq is None:
        freq = 1 / duration

    return _sampled_sin_pulse(duration, amp, freq, phase=phase, name=name)


_sampled_gaussian_pulse = samplers.midpoint(continuous.gaussian)


def gaussian(
    duration: int, amp: complex, sigma: float, name: Optional[str] = None, zero_ends: bool = True
) -> Waveform:
    r"""Generates unnormalized gaussian :class:`~qiskit.pulse.Waveform`.

    For :math:`A=` ``amp`` and :math:`\sigma=` ``sigma``, applies the ``midpoint`` sampling strategy
    to generate a discrete pulse sampled from the continuous function:

    .. math::

        f(x) = A\exp\left(\left(\frac{x - \mu}{2\sigma}\right)^2 \right),

    with the center :math:`\mu=` ``duration/2``.

    If ``zero_ends==True``, each output sample :math:`y` is modified according to:

    .. math::

        y \mapsto A\frac{y-y^*}{A-y^*},

    where :math:`y^*` is the value of the endpoint samples. This sets the endpoints
    to :math:`0` while preserving the amplitude at the center. If :math:`A=y^*`,
    :math:`y` is set to :math:`1`. By default, the endpoints are at ``x = -1, x = duration + 1``.

    Integrated area under the full curve is ``amp * np.sqrt(2*np.pi*sigma**2)``

    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Pulse amplitude at ``duration/2``.
        sigma: Width (standard deviation) of pulse.
        name: Name of pulse.
        zero_ends: If True, zero ends at ``x = -1, x = duration + 1``, but rescale to preserve amp.
    """
    center = duration / 2
    zeroed_width = duration + 2 if zero_ends else None
    rescale_amp = bool(zero_ends)
    return _sampled_gaussian_pulse(
        duration, amp, center, sigma, zeroed_width=zeroed_width, rescale_amp=rescale_amp, name=name
    )


_sampled_gaussian_deriv_pulse = samplers.midpoint(continuous.gaussian_deriv)


def gaussian_deriv(
    duration: int, amp: complex, sigma: float, name: Optional[str] = None
) -> Waveform:
    r"""Generates unnormalized gaussian derivative :class:`~qiskit.pulse.Waveform`.

    For :math:`A=` ``amp`` and :math:`\sigma=` ``sigma`` applies the `midpoint` sampling strategy
    to generate a discrete pulse sampled from the continuous function:

    .. math::

        f(x) = A\frac{(x - \mu)}{\sigma^2}\exp\left(\left(\frac{x - \mu}{2\sigma}\right)^2 \right)

    i.e. the derivative of the Gaussian function, with center :math:`\mu=` ``duration/2``.

    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Pulse amplitude of corresponding Gaussian at the pulse center (``duration/2``).
        sigma: Width (standard deviation) of pulse.
        name: Name of pulse.
    """
    center = duration / 2
    return _sampled_gaussian_deriv_pulse(duration, amp, center, sigma, name=name)


_sampled_sech_pulse = samplers.midpoint(continuous.sech)


def sech(
    duration: int, amp: complex, sigma: float, name: str = None, zero_ends: bool = True
) -> Waveform:
    r"""Generates unnormalized sech :class:`~qiskit.pulse.Waveform`.

    For :math:`A=` ``amp`` and :math:`\sigma=` ``sigma``, applies the ``midpoint`` sampling strategy
    to generate a discrete pulse sampled from the continuous function:

    .. math::

        f(x) = A\text{sech}\left(\frac{x-\mu}{\sigma} \right)

    with the center :math:`\mu=` ``duration/2``.

    If ``zero_ends==True``, each output sample :math:`y` is modified according to:

    .. math::

        y \mapsto A\frac{y-y^*}{A-y^*},

    where :math:`y^*` is the value of the endpoint samples. This sets the endpoints
    to :math:`0` while preserving the amplitude at the center. If :math:`A=y^*`,
    :math:`y` is set to :math:`1`. By default, the endpoints are at ``x = -1, x = duration + 1``.

    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Pulse amplitude at `duration/2`.
        sigma: Width (standard deviation) of pulse.
        name: Name of pulse.
        zero_ends: If True, zero ends at ``x = -1, x = duration + 1``, but rescale to preserve amp.
    """
    center = duration / 2
    zeroed_width = duration + 2 if zero_ends else None
    rescale_amp = bool(zero_ends)
    return _sampled_sech_pulse(
        duration, amp, center, sigma, zeroed_width=zeroed_width, rescale_amp=rescale_amp, name=name
    )


_sampled_sech_deriv_pulse = samplers.midpoint(continuous.sech_deriv)


def sech_deriv(duration: int, amp: complex, sigma: float, name: str = None) -> Waveform:
    r"""Generates unnormalized sech derivative :class:`~qiskit.pulse.Waveform`.

    For :math:`A=` ``amp``, :math:`\sigma=` ``sigma``, and center :math:`\mu=` ``duration/2``,
    applies the `midpoint` sampling strategy to generate a discrete pulse sampled from
    the continuous function:

    .. math::
        f(x) = \frac{d}{dx}\left[A\text{sech}\left(\frac{x-\mu}{\sigma} \right)\right],

    i.e. the derivative of :math:`\text{sech}`.

    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Pulse amplitude at `center`.
        sigma: Width (standard deviation) of pulse.
        name: Name of pulse.
    """
    center = duration / 2
    return _sampled_sech_deriv_pulse(duration, amp, center, sigma, name=name)


_sampled_gaussian_square_pulse = samplers.midpoint(continuous.gaussian_square)


def gaussian_square(
    duration: int,
    amp: complex,
    sigma: float,
    risefall: Optional[float] = None,
    width: Optional[float] = None,
    name: Optional[str] = None,
    zero_ends: bool = True,
) -> Waveform:
    r"""Generates gaussian square :class:`~qiskit.pulse.Waveform`.

    For :math:`d=` ``duration``, :math:`A=` ``amp``, :math:`\sigma=` ``sigma``,
    and :math:`r=` ``risefall``, applies the ``midpoint`` sampling strategy to
    generate a discrete pulse sampled from the continuous function:

    .. math::

        f(x) = \begin{cases}
                    g(x - r) ) & x\leq r \\
                    A & r\leq x\leq d-r \\
                    g(x - (d - r)) & d-r\leq x
                \end{cases}

    where :math:`g(x)` is the Gaussian function sampled from in :meth:`gaussian`
    with :math:`A=` ``amp``, :math:`\mu=1`, and :math:`\sigma=` ``sigma``. I.e.
    :math:`f(x)` represents a square pulse with smooth Gaussian edges.

    If ``zero_ends == True``, the samples for the Gaussian ramps are remapped as in
    :meth:`gaussian`.

    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Pulse amplitude.
        sigma: Width (standard deviation) of Gaussian rise/fall portion of the pulse.
        risefall: Number of samples over which pulse rise and fall happen. Width of
            square portion of pulse will be ``duration-2*risefall``.
        width: The duration of the embedded square pulse. Only one of ``width`` or ``risefall``
               should be specified as the functional form requires
               ``width = duration - 2 * risefall``.
        name: Name of pulse.
        zero_ends: If True, zero ends at ``x = -1, x = duration + 1``, but rescale to preserve amp.

    Raises:
        PulseError: If ``risefall`` and ``width`` arguments are inconsistent or not enough info.
    """
    if risefall is None and width is None:
        raise PulseError("gaussian_square missing required argument: 'width' or 'risefall'.")
    if risefall is not None:
        if width is None:
            width = duration - 2 * risefall
        elif 2 * risefall + width != duration:
            raise PulseError(
                "Both width and risefall were specified, and they are "
                "inconsistent: 2 * risefall + width == {} != "
                "duration == {}.".format(2 * risefall + width, duration)
            )
    center = duration / 2
    zeroed_width = duration + 2 if zero_ends else None
    return _sampled_gaussian_square_pulse(
        duration, amp, center, width, sigma, zeroed_width=zeroed_width, name=name
    )


_sampled_drag_pulse = samplers.midpoint(continuous.drag)


def drag(
    duration: int,
    amp: complex,
    sigma: float,
    beta: float,
    name: Optional[str] = None,
    zero_ends: bool = True,
) -> Waveform:
    r"""Generates Y-only correction DRAG :class:`~qiskit.pulse.Waveform` for standard nonlinear
    oscillator (SNO) [1].

    For :math:`A=` ``amp``, :math:`\sigma=` ``sigma``, and :math:`\beta=` ``beta``, applies the
    ``midpoint`` sampling strategy to generate a discrete pulse sampled from the
    continuous function:

    .. math::

        f(x) = g(x) + i \beta h(x),

    where :math:`g(x)` is the function sampled in :meth:`gaussian`, and :math:`h(x)`
    is the function sampled in :meth:`gaussian_deriv`.

    If ``zero_ends == True``, the samples from :math:`g(x)` are remapped as in :meth:`gaussian`.

    References:
        1. |citation1|_

        .. _citation1: http://dx.doi.org/10.1103/PhysRevA.83.012308

        .. |citation1| replace:: *Gambetta, J. M., Motzoi, F., Merkel, S. T. & Wilhelm, F. K.
           "Analytic control methods for high-fidelity unitary operations
           in a weakly nonlinear oscillator." Phys. Rev. A 83, 012308 (2011).*

    Args:
        duration: Duration of pulse. Must be greater than zero.
        amp: Pulse amplitude at center  ``duration/2``.
        sigma: Width (standard deviation) of pulse.
        beta: Y correction amplitude. For the SNO this is
              :math:`\beta=-\frac{\lambda_1^2}{4\Delta_2}`. Where :math:`\lambda_1` is the
              relative coupling strength between the first excited and second excited states
              and :math:`\Delta_2` is the detuning between the respective excited states.
        name: Name of pulse.
        zero_ends: If True, zero ends at ``x = -1, x = duration + 1``, but rescale to preserve amp.
    """
    center = duration / 2
    zeroed_width = duration + 2 if zero_ends else None
    rescale_amp = bool(zero_ends)
    return _sampled_drag_pulse(
        duration,
        amp,
        center,
        sigma,
        beta,
        zeroed_width=zeroed_width,
        rescale_amp=rescale_amp,
        name=name,
    )
