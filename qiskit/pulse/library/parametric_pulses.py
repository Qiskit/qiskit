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

"""Parametric waveforms module. These are pulses which are described by a specified
parameterization.

If a backend supports parametric pulses, it will have the attribute
`backend.configuration().parametric_pulses`, which is a list of supported pulse shapes, such as
`['gaussian', 'gaussian_square', 'drag']`. A Pulse Schedule, using parametric pulses, which is
assembled for a backend which supports those pulses, will result in a Qobj which is dramatically
smaller than one which uses Waveforms.

This module can easily be extended to describe more pulse shapes. The new class should:
  - have a descriptive name
  - be a well known and/or well described formula (include the formula in the class docstring)
  - take some parameters (at least `duration`) and validate them, if necessary
  - implement a ``get_waveform`` method which returns a corresponding Waveform in the case that
    it is assembled for a backend which does not support it. Ends are zeroed to avoid steep jumps at
    pulse edges. By default, the ends are defined such that ``f(-1), f(duration+1) = 0``.

The new pulse must then be registered by the assembler in
`qiskit/qobj/converters/pulse_instruction.py:ParametricPulseShapes`
by following the existing pattern:

    class ParametricPulseShapes(Enum):
        gaussian = library.Gaussian
        ...
        new_supported_pulse_name = library.YourPulseWaveformClass
"""
from abc import abstractmethod
from typing import Any, Dict, Tuple, Optional, Union

import math
import numpy as np

from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.library import continuous, symbolic
from qiskit.pulse.library.pulse import Pulse
from qiskit.pulse.library.waveform import Waveform
from qiskit.pulse.utils import deprecated_functionality
from qiskit.utils import optionals


class ParametricPulse(Pulse):
    """The abstract superclass for parametric pulses."""
    __slots__ = ("param_values", "definition")

    PARAM_DEF = ["duration"]

    @abstractmethod
    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        parameters: Optional[Tuple[Union[ParameterExpression, complex]]] = None,
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
    ):
        """Create a parametric pulse and validate the input parameters.

        Args:
            duration: Pulse length in terms of the the sampling period `dt`.
            parameters: Other parameters to form waveform.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                             waveform to 1. The default is ``True`` and the
                             amplitude is constrained to 1.
        """
        super().__init__(duration=duration, name=name, limit_amplitude=limit_amplitude)

        if parameters:
            self.param_values = (duration, ) + tuple(parameters)
        else:
            self.param_values = (duration, )

        self.definition = self._define()
        self.validate_parameters()

    def _define(self):
        """Return symbolic expression of pulse waveform.

        A custom pulse without having this method implemented cannot be QPY serialized.
        The subclass must override :meth:`get_waveform` method to return waveform.
        Note that the expression should contain symbol ``t`` that represents a
        sampling time, along with parameters defined in :meth:`.parameters`.
        """
        return None

    def get_waveform(self) -> Waveform:
        r"""Return a Waveform with samples filled according to the formula that the pulse
        represents and the parameter values it contains.

        Since the returned array is discretized time series of the continuous function,
        this method uses midpoint sampler. For ``duration``, return:

        .. math::

            \{f(t+0.5) \in \mathbb{C} | t \in \mathbb{Z} \wedge  0<=t<\texttt{duration}\}

        Returns:
            A waveform representation of this pulse.

        Raises:
            PulseError: When parameters are not bound.
        """
        if self.is_parameterized():
            raise PulseError(
                f"Unassigned parameter exists: {self.parameters}. All parameters should be assigned."
            )

        if optionals.HAS_SYMENGINE:
            import symengine as sym
        else:
            import sympy as sym

        params = {sym.Symbol(k): v for k, v in self.parameters.items()}
        t = sym.Symbol("t")
        assigned_expr = self.definition.subs(params)

        # midpoint sampling
        times = np.arange(0, self.duration) + 1/2
        if optionals.HAS_SYMENGINE:
            lambda_func = sym.lambdify(t, [assigned_expr.simplify()])
        else:
            # lambda_func = sym.lambdify(t, assigned_expr.simplify())
            lambda_func = sym.lambdify(t, assigned_expr)

        waveform = lambda_func(times)
        if not isinstance(waveform, np.ndarray):
            # Workaround for constants being simplified to scalars
            # See: https://github.com/sympy/sympy/issues/5642
            waveform = waveform * np.ones(times.shape)

        # this should be simplified to handle piecewise function

        return Waveform(samples=waveform, name=self.name)

    @abstractmethod
    def validate_parameters(self) -> None:
        """
        Validate parameters.

        Raises:
            PulseError: If the parameters passed are not valid.
        """
        raise NotImplementedError

    def is_parameterized(self) -> bool:
        """Return True iff the instruction is parameterized."""
        return any(_is_parameterized(val) for val in self.param_values)

    @property
    def parameters(self) -> Dict[str, Any]:
        return dict(zip(self.PARAM_DEF, self.param_values))

    def __eq__(self, other: Pulse) -> bool:
        return super().__eq__(other) and self.parameters == other.parameters

    def __hash__(self) -> int:
        return hash(tuple(self.parameters[k] for k in sorted(self.parameters)))

    def __repr__(self) -> str:
        params_str = ", ".join(f"{p}={v}" for p, v in zip(self.PARAM_DEF, self.param_values))

        return "{}({}{})".format(
            self.__class__.__name__,
            params_str,
            f", name='{self.name}'" if self.name is not None else "",
        )


class Gaussian(ParametricPulse):
    r"""A lifted and truncated pulse envelope shaped according to the Gaussian function whose
    mean is centered at the center of the pulse (duration / 2):

    .. math::

        f'(x) &= \exp\Bigl( -\frac12 \frac{{(x - \text{duration}/2)}^2}{\text{sigma}^2} \Bigr)\\
        f(x) &= \text{amp} \times \frac{f'(x) - f'(-1)}{1-f'(-1)}, \quad 0 \le x < \text{duration}

    where :math:`f'(x)` is the gaussian waveform without lifting or amplitude scaling.

    This pulse would be more accurately named as ``LiftedGaussian``, however, for historical
    and practical DSP reasons it has the name ``Gaussian``.
    """
    PARAM_DEF = ["duration", "amp", "sigma"]

    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, ParameterExpression],
        sigma: Union[float, ParameterExpression],
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
    ):
        """Initialize the gaussian pulse.

        Args:
            duration: Pulse length in terms of the the sampling period `dt`.
            amp: The amplitude of the Gaussian envelope.
            sigma: A measure of how wide or narrow the Gaussian peak is; described mathematically
                   in the class docstring.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                             waveform to 1. The default is ``True`` and the
                             amplitude is constrained to 1.
        """
        if not _is_parameterized(amp):
            amp = complex(amp)

        super().__init__(
            duration=duration,
            parameters=(amp, sigma),
            name=name,
            limit_amplitude=limit_amplitude,
        )

    @property
    @deprecated_functionality
    def amp(self) -> Union[complex, ParameterExpression]:
        """The Gaussian amplitude."""
        return self.param_values[1]

    @property
    @deprecated_functionality
    def sigma(self) -> Union[float, ParameterExpression]:
        """The Gaussian standard deviation of the pulse width."""
        return self.param_values[2]

    def _define(self):
        return symbolic.symbolic_gaussian()

    def validate_parameters(self) -> None:
        _, amp, sigma = self.param_values

        if not _is_parameterized(amp) and abs(amp) > 1.0 and self.limit_amplitude:
            raise PulseError(
                f"The amplitude norm must be <= 1, found: {abs(amp)}"
                + "This can be overruled by setting Pulse.limit_amplitude."
            )
        if not _is_parameterized(sigma) and sigma <= 0:
            raise PulseError("Sigma must be greater than 0.")


class GaussianSquare(ParametricPulse):
    # Not a raw string because we need to be able to split lines.
    """A square pulse with a Gaussian shaped risefall on both sides lifted such that
    its first sample is zero.

    Either the ``risefall_sigma_ratio`` or ``width`` parameter has to be specified.

    If ``risefall_sigma_ratio`` is not None and ``width`` is None:

    .. math::

        \\text{risefall} &= \\text{risefall_sigma_ratio} \\times \\text{sigma}\\\\
        \\text{width} &= \\text{duration} - 2 \\times \\text{risefall}

    If ``width`` is not None and ``risefall_sigma_ratio`` is None:

    .. math:: \\text{risefall} = \\frac{\\text{duration} - \\text{width}}{2}

    In both cases, the lifted gaussian square pulse :math:`f'(x)` is defined as:

    .. math::

        f'(x) &= \\begin{cases}\
            \\exp\\biggl(-\\frac12 \\frac{(x - \\text{risefall})^2}{\\text{sigma}^2}\\biggr)\
                & x < \\text{risefall}\\\\
            1\
                & \\text{risefall} \\le x < \\text{risefall} + \\text{width}\\\\
            \\exp\\biggl(-\\frac12\
                    \\frac{{\\bigl(x - (\\text{risefall} + \\text{width})\\bigr)}^2}\
                          {\\text{sigma}^2}\
                    \\biggr)\
                & \\text{risefall} + \\text{width} \\le x\
        \\end{cases}\\\\
        f(x) &= \\text{amp} \\times \\frac{f'(x) - f'(-1)}{1-f'(-1)},\
            \\quad 0 \\le x < \\text{duration}

    where :math:`f'(x)` is the gaussian square waveform without lifting or amplitude scaling.

    This pulse would be more accurately named as ``LiftedGaussianSquare``, however, for historical
    and practical DSP reasons it has the name ``GaussianSquare``.
    """
    __slots__ = ("_risefall_sigma_ratio", )

    PARAM_DEF = ["duration", "amp", "sigma", "width"]

    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, ParameterExpression],
        sigma: Union[float, ParameterExpression],
        width: Union[float, ParameterExpression] = None,
        risefall_sigma_ratio: Union[float, ParameterExpression] = None,
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
    ):
        """Initialize the gaussian square pulse.

        Args:
            duration: Pulse length in terms of the the sampling period `dt`.
            amp: The amplitude of the Gaussian and of the square pulse.
            sigma: A measure of how wide or narrow the Gaussian risefall is; see the class
                   docstring for more details.
            width: The duration of the embedded square pulse.
            risefall_sigma_ratio: The ratio of each risefall duration to sigma.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                             waveform to 1. The default is ``True`` and the
                             amplitude is constrained to 1.
        """
        if not _is_parameterized(amp):
            amp = complex(amp)

        # Convert risefall_sigma_ratio into width which is defined in OpenPulse spec
        if width is None and risefall_sigma_ratio is None:
            raise PulseError(
                "Either the pulse width or the risefall_sigma_ratio parameter must be specified."
            )
        if width is None:
            width = duration - 2.0 * risefall_sigma_ratio * sigma
        else:
            risefall_sigma_ratio = (duration - width) / (2.0 * sigma)

        self._risefall_sigma_ratio = risefall_sigma_ratio
        super().__init__(
            duration=duration,
            parameters=(amp, sigma, width),
            name=name,
            limit_amplitude=limit_amplitude
        )

    @property
    @deprecated_functionality
    def amp(self) -> Union[complex, ParameterExpression]:
        """The Gaussian amplitude."""
        return self.param_values[1]

    @property
    @deprecated_functionality
    def sigma(self) -> Union[float, ParameterExpression]:
        """The Gaussian standard deviation of the pulse width."""
        return self.param_values[2]

    @property
    @deprecated_functionality
    def risefall_sigma_ratio(self) -> Union[float, ParameterExpression]:
        """The duration of each risefall in terms of sigma."""
        return self._risefall_sigma_ratio

    @property
    @deprecated_functionality
    def width(self) -> Union[float, ParameterExpression]:
        """The width of the square portion of the pulse."""
        return self.param_values[3]

    def _define(self):
        return symbolic.symbolic_gaussian_square()

    def validate_parameters(self) -> None:
        duration, amp, sigma, width = self.param_values

        if not _is_parameterized(amp) and abs(amp) > 1.0 and self.limit_amplitude:
            raise PulseError(
                f"The amplitude norm must be <= 1, found: {abs(amp)}"
                + "This can be overruled by setting Pulse.limit_amplitude."
            )
        if not _is_parameterized(sigma) and sigma <= 0:
            raise PulseError("Sigma must be greater than 0.")

        if not _is_parameterized(width) and width < 0:
            raise PulseError("The pulse width must be at least 0.")
        if (
            not (_is_parameterized(width) or _is_parameterized(duration))
            and width >= duration
        ):
            raise PulseError(
                "The pulse width must be less than its duration, or "
                "the parameter risefall_sigma_ratio must be less than duration/(2*sigma)."
            )


class Drag(ParametricPulse):
    """The Derivative Removal by Adiabatic Gate (DRAG) pulse is a standard Gaussian pulse
    with an additional Gaussian derivative component and lifting applied.

    It is designed to reduce the frequency spectrum of a normal gaussian pulse near
    the :math:`|1\\rangle\\leftrightarrow|2\\rangle` transition,
    reducing the chance of leakage to the :math:`|2\\rangle` state.

    .. math::

        g(x) &= \\exp\\Bigl(-\\frac12 \\frac{(x - \\text{duration}/2)^2}{\\text{sigma}^2}\\Bigr)\\\\
        f'(x) &= g(x) + 1j \\times \\text{beta} \\times \\frac{\\mathrm d}{\\mathrm{d}x} g(x)\\\\
              &= g(x) + 1j \\times \\text{beta} \\times\
                    \\Bigl(-\\frac{x - \\text{duration}/2}{\\text{sigma}^2}\\Bigr)g(x)\\\\
        f(x) &= \\text{amp}\\times\\frac{f'(x)-f'(-1)}{1-f'(-1)}, \\quad 0 \\le x < \\text{duration}

    where :math:`g(x)` is a standard unlifted gaussian waveform and
    :math:`f'(x)` is the DRAG waveform without lifting or amplitude scaling.

    This pulse would be more accurately named as ``LiftedDrag``, however, for historical
    and practical DSP reasons it has the name ``Drag``.

    References:
        1. |citation1|_

        .. _citation1: https://link.aps.org/doi/10.1103/PhysRevA.83.012308

        .. |citation1| replace:: *Gambetta, J. M., Motzoi, F., Merkel, S. T. & Wilhelm, F. K.
           Analytic control methods for high-fidelity unitary operations
           in a weakly nonlinear oscillator. Phys. Rev. A 83, 012308 (2011).*

        2. |citation2|_

        .. _citation2: https://link.aps.org/doi/10.1103/PhysRevLett.103.110501

        .. |citation2| replace:: *F. Motzoi, J. M. Gambetta, P. Rebentrost, and F. K. Wilhelm
           Phys. Rev. Lett. 103, 110501 – Published 8 September 2009.*
    """
    PARAM_DEF = ["duration", "amp", "sigma", "beta"]

    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, ParameterExpression],
        sigma: Union[float, ParameterExpression],
        beta: Union[float, ParameterExpression],
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
    ):
        """Initialize the drag pulse.

        Args:
            duration: Pulse length in terms of the the sampling period `dt`.
            amp: The amplitude of the Drag envelope.
            sigma: A measure of how wide or narrow the Gaussian peak is; described mathematically
                   in the class docstring.
            beta: The correction amplitude.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                             waveform to 1. The default is ``True`` and the
                             amplitude is constrained to 1.
        """
        if not _is_parameterized(amp):
            amp = complex(amp)

        super().__init__(
            duration=duration,
            parameters=(amp, sigma, beta),
            name=name,
            limit_amplitude=limit_amplitude,
        )

    @property
    @deprecated_functionality
    def amp(self) -> Union[complex, ParameterExpression]:
        """The Gaussian amplitude."""
        return self.param_values[1]

    @property
    @deprecated_functionality
    def sigma(self) -> Union[float, ParameterExpression]:
        """The Gaussian standard deviation of the pulse width."""
        return self.param_values[2]

    @property
    @deprecated_functionality
    def beta(self) -> Union[float, ParameterExpression]:
        """The weighing factor for the Gaussian derivative component of the waveform."""
        return self.param_values[3]

    def _define(self):
        return symbolic.symbolic_drag()

    def validate_parameters(self) -> None:
        duration, amp, sigma, beta = self.param_values

        if not _is_parameterized(amp) and abs(amp) > 1.0 and self.limit_amplitude:
            raise PulseError(
                f"The amplitude norm must be <= 1, found: {abs(amp)}"
                + "This can be overruled by setting Pulse.limit_amplitude."
            )
        if not _is_parameterized(sigma) and sigma <= 0:
            raise PulseError("Sigma must be greater than 0.")
        if not _is_parameterized(beta) and isinstance(beta, complex):
            raise PulseError("Beta must be real.")
        # Check if beta is too large: the amplitude norm must be <=1 for all points
        if (
            not _is_parameterized(beta)
            and not _is_parameterized(sigma)
            and beta > sigma
            and self.limit_amplitude
        ):
            # If beta <= sigma, then the maximum amplitude is at duration / 2, which is
            # already constrained by amp <= 1

            # 1. Find the first maxima associated with the beta * d/dx gaussian term
            #    This eq is derived from solving for the roots of the norm of the drag function.
            #    There is a second maxima mirrored around the center of the pulse with the same
            #    norm as the first, so checking the value at the first x maxima is sufficient.
            argmax_x = duration / 2 - (sigma / beta) * math.sqrt(beta**2 - sigma**2)
            # If the max point is out of range, either end of the pulse will do
            argmax_x = max(argmax_x, 0)

            # 2. Find the value at that maximum
            max_val = continuous.drag(
                np.array(argmax_x),
                sigma=sigma,
                beta=beta,
                amp=amp,
                center=duration / 2,
            )
            if abs(max_val) > 1.0:
                raise PulseError("Beta is too large; pulse amplitude norm exceeds 1.")


class Constant(ParametricPulse):
    """
    A simple constant pulse, with an amplitude value and a duration:

    .. math::

        f(x) = amp    ,  0 <= x < duration
        f(x) = 0      ,  elsewhere
    """
    PARAM_DEF = ["duration", "amp"]

    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, ParameterExpression],
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
    ):
        """
        Initialize the constant-valued pulse.

        Args:
            duration: Pulse length in terms of the the sampling period `dt`.
            amp: The amplitude of the constant square pulse.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                             waveform to 1. The default is ``True`` and the
                             amplitude is constrained to 1.
        """
        if not _is_parameterized(amp):
            amp = complex(amp)

        super().__init__(
            duration=duration,
            parameters=(amp, ),
            name=name,
            limit_amplitude=limit_amplitude,
        )

    @property
    @deprecated_functionality
    def amp(self) -> Union[complex, ParameterExpression]:
        """The constant value amplitude."""
        return self.param_values[1]

    def _define(self):
        return symbolic.symbolic_constant()

    def validate_parameters(self) -> None:
        amp = self.param_values[1]
        if not _is_parameterized(amp) and abs(amp) > 1.0 and self.limit_amplitude:
            raise PulseError(
                f"The amplitude norm must be <= 1, found: {abs(amp)}"
                + "This can be overruled by setting Pulse.limit_amplitude."
            )


def _is_parameterized(value: Any) -> bool:
    """Shorthand for a frequently checked predicate. ParameterExpressions cannot be
    validated until they are numerically assigned.
    """
    return isinstance(value, ParameterExpression)
