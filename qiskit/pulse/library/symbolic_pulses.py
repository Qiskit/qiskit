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

# pylint: disable=invalid-name

"""Symbolic waveform module.

These are pulses which are described by symbolic equations for envelope and parameter constraints.
"""

import functools
from typing import Any, Dict, List, Optional, Union, Callable, TYPE_CHECKING

import numpy as np

from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.library.pulse import Pulse
from qiskit.pulse.library.waveform import Waveform

if TYPE_CHECKING:
    # Note that the symbolic pulse module doesn't employ symengine due to some missing features.
    # In addition, there are several syntactic difference in boolean expression.
    # Thanks to Lambdify at subclass instantiation, the performance regression is not significant.
    # However, this will still create significant latency for the first sympy import.
    from sympy import Expr, Symbol


def _lifted_gaussian(
    t: "Symbol",
    center: Union["Symbol", "Expr", complex],
    t_zero: Union["Symbol", "Expr", complex],
    sigma: Union["Symbol", "Expr", complex],
) -> "Expr":
    r"""Helper function that returns a lifted Gaussian symbolic equation.

    For :math:`\sigma=` ``sigma`` the symbolic equation will be

    .. math::

        f(x) = \exp\left(-\frac12 \left(\frac{x - \mu}{\sigma}\right)^2 \right),

    with the center :math:`\mu=` ``duration/2``.
    Then, each output sample :math:`y` is modified according to:

    .. math::

        y \mapsto \frac{y-y^*}{1.0-y^*},

    where :math:`y^*` is the value of the un-normalized Gaussian at the endpoints of the pulse.
    This sets the endpoints to :math:`0` while preserving the amplitude at the center,
    i.e. :math:`y` is set to :math:`1.0`.

    Args:
        t: Symbol object representing time.
        center: Symbol or expression representing the middle point of the samples.
        t_zero: The value of t at which the pulse is lowered to 0.
        sigma: Symbol or expression representing Gaussian sigma.

    Returns:
        Symbolic equation.
    """
    import sympy as sym

    gauss = sym.exp(-(((t - center) / sigma) ** 2) / 2)
    offset = sym.exp(-(((t_zero - center) / sigma) ** 2) / 2)

    return (gauss - offset) / (1 - offset)


@functools.lru_cache(maxsize=None)
def _validate_amplitude_limit(symbolic_pulse: "SymbolicPulse") -> bool:
    """A helper function to validate maximum amplitude limit.

    Result is cached for better performance.

    Args:
        symbolic_pulse: A pulse to validate.

    Returns:
        Return True if any sample point exceeds 1.0 in absolute value.
    """
    return np.any(np.abs(symbolic_pulse.get_waveform().samples) > 1.0)


class LamdifiedExpression:
    """Descriptor to lambdify symbolic expression with cache.

    When new symbolic expression is set for the first time,
    this will internally lambdify the expressions and store the callbacks in the instance cache.
    For the next time it will just return the cached callbacks for speed up.
    """

    def __init__(self, common_params: List[str], attribute: str):
        """Create new descriptor.

        Args:
            common_params: Parameters that the expression always takes regardless
                of pulse instances.
            attribute: Name of attribute of :class:`.SymbolicPulse` that returns
                the target expression to evaluate.
        """
        self.common_params = common_params
        self.attribute = attribute
        self.lambda_funcs = dict()

    def __get__(self, instance, owner) -> Callable:
        expr = getattr(instance, self.attribute, None)
        key = hash(expr)
        if key not in self.lambda_funcs:
            if expr is None:
                raise PulseError(f"'{self.attribute}' of '{instance.pulse_type}' is not assigned.")
            self.__set__(instance, expr)

        return self.lambda_funcs[key]

    def __set__(self, instance, value):
        key = hash(value)
        if key not in self.lambda_funcs:
            import sympy as sym

            params = [sym.Symbol(p) for p in self.common_params + instance._param_names]
            self.lambda_funcs[key] = sym.lambdify(params, value)


class SymbolicPulse(Pulse):
    """The pulse representation model with parameters and symbolic expressions.

    A symbolic pulse instance can be defined with an envelope and parameter constraints.
    Envelope and parameter constraints should be provided with symbolic expressions.
    Rather than creating a subclass, different pulse shapes can be distinguished by
    the instance attributes :attr:`SymbolicPulse.envelope` and :attr:`SymbolicPulse.constraints`,
    together with the ``pulse_type`` argument of the :class:`SymbolicPulse` constructor.


    .. _symbolic_pulse_envelope:

    .. rubric:: Envelope function

    This is defined with an instance attribute :attr:`SymbolicPulse.envelope`
    which can be provided through its setter method.
    The expression must be a function of ``t``, ``duration``, ``amp`` and
    any additional parameters specified for the pulse shape to implement.
    The time ``t`` and ``duration`` are in units of dt, i.e. sample time resolution,
    and this function is sampled with a discrete time vector in [0, ``duration``]
    sampling the pulse envelope at every 0.5 dt (middle sampling strategy) when
    :meth:`SymbolicPulse.get_waveform` method is called.
    The ``amp`` is a complex-valued coefficient that scales the symbolic pulse envelope.
    This indicates conventionally the Qiskit Pulse conforms to the IQ format rather
    than the phasor representation. When a real value is assigned to the ``amp``,
    it is internally typecasted to the complex. The real and imaginary part may be
    directly supplied to two quadratures of the IQ mixer in the control electronics.
    The sample data is not generated until the :meth:`SymbolicPulse.get_waveform` method is called
    thus a symbolic pulse instance only stores parameter values and waveform shape,
    which greatly reduces memory footprint during the program generation.


    .. _symbolic_pulse_constraints:

    .. rubric:: Constraint functions

    Constraints on the parameters are defined with an instance attribute
    :attr:`SymbolicPulse.constraints` which can be provided through its setter method.
    The constraints value must be a symbolic expression, which is a
    function of parameters to be validated and must return a boolean value
    being ``True`` when parameters are valid.
    If there are multiple conditions to be evaluated, these conditions can be
    concatenated with logical expressions such as ``sym.And`` and ``sym.Or``.
    The symbolic pulse instance can be played only when the constraint function returns ``True``.
    The constraint is evaluated when a :meth:`SymbolicPulse.validate_parameters` is called.
    Note that the maximum pulse amplitude limit is separately evaluated when
    the :attr:`.limit_amplitude` is set.
    Since this is evaluated with actual waveform samples by calling :meth:`.get_waveform`,
    it is not necessary to define any explicit constraint for the amplitude limitation.

    .. rubric:: Examples

    This is how user can instantiate symbolic pulse instance.
    In this example, we instantiate a custom `Sawtooth` envelope.

    .. jupyter-execute::

        from qiskit.pulse.library import SymbolicPulse

        my_pulse = SymbolicPulse(
            pulse_type="Sawtooth",
            parameters={"duration": 100, "amp": 0.1, "freq": 0.05},
            name="pulse1",
        )

    Note that :class:`SymbolicPulse` can be instantiated without providing
    the envelope and constraints. However, this instance cannot generate waveforms
    without knowing the envelope definition. Now you need to provide the envelope.

    .. jupyter-execute::

        import sympy

        t, amp, freq = sympy.symbols("t, amp, freq")
        envelope = amp * 2 * (freq * t - sympy.floor(1 / 2 + freq * t))
        my_pulse.envelope = envelope

        my_pulse.draw()

    Likewise, you can define :attr:`SymbolicPulse.constraints` for ``my_pulse``.
    After providing the envelope definition, you can generate the waveform data.
    Note that it would be convenient to define a factory function that automatically
    accomplishes this procedure.

    .. code-block:: python

        def Sawtooth(duration, amp, freq, name):
            instance = SymbolicPulse(
                pulse_type="Sawtooth",
                parameters={"duration": duration, "amp": amp, "freq": freq},
                name=name,
            )

            t, amp, freq = sympy.symbols("t, amp, freq")
            instance.envelope = amp * 2 * (freq * t - sympy.floor(1 / 2 + freq * t))

            return instance

    You can also provide a :class:`Parameter` object in the ``parameters`` dictionary
    when you instantiate the symbolic pulse instance. Waveform cannot be
    generated until you assign all unbounded parameters.
    Note that parameters will be assigned through the schedule playing the pulse.


    .. _symbolic_pulse_serialize:

    .. rubric:: Serialization

    The :class:`~SymbolicPulse` subclass is QPY serialized with symbolic expressions.
    A user can therefore create a custom pulse subclass with a novel envelope and constraints,
    and then one can instantiate the class with certain parameters to run on a backend.
    This pulse instance can be saved in the QPY binary, which can be loaded afterwards
    even within the environment not having original class definition loaded.
    This mechanism allows us to easily share a pulse program including custom pulse instructions
    with collaborators or to directly submit the program to the quantum computer backend
    in the parametric form (i.e. not sample data). This greatly reduces amount of data transferred.
    The waveform sample data to be loaded in the control electronics
    will be reconstructed with parameters and deserialized expression objects by the backend.

    .. note::

        Currently QPY serialization of :class:`SymbolicPulse` is not available.
        This feature will be implemented shortly.
        Note that data transmission in the parametric form requires your quantum computer backend
        to support QPY framework with :class:`SymbolicPulse` available. Otherwise, the pulse data
        might be converted into :class:`Waveform`.
    """

    # Lambdify caches keyed on sympy expressions. Returns the corresponding callable.
    _callable_envelope = LamdifiedExpression(
        common_params=["t", "duration", "amp"],
        attribute="_envelope_expr",
    )
    _callable_consts = LamdifiedExpression(
        common_params=["duration", "amp"],
        attribute="_consts_expr",
    )

    def __init__(
        self,
        pulse_type: str,
        parameters: Dict[str, Union[ParameterExpression, complex]],
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
    ):
        """Create a parametric pulse and validate the input parameters.

        Args:
            pulse_type: Display name of this pulse shape.
            parameters: Dictionary of pulse parameters. This must include "duration".
            name: Display name for this particular pulse envelope.
            limit_amplitude: If ``True``, then limit the absolute value of the amplitude of the
                waveform to 1. The default is ``True`` and the amplitude is constrained to 1.

        Raises:
            PulseError: When not all parameters are listed in the attribute :attr:`PARAM_DEF`.
        """
        if "duration" not in parameters:
            raise PulseError("'duration' is not defined for this pulse.")
        if "amp" not in parameters:
            raise PulseError("'amp' is not defined for this pulse.")

        super().__init__(
            duration=parameters.pop("duration"),
            name=name,
            limit_amplitude=limit_amplitude,
        )
        amp = parameters.pop("amp")
        if not isinstance(amp, ParameterExpression):
            amp = complex(amp)
        self.amp = amp

        self._pulse_type = pulse_type
        self._param_names = list(parameters.keys())
        self._param_vals = list(parameters.values())
        self._envelope_expr = None
        self._consts_expr = None

    def __getattr__(self, item):
        # For backward compatibility. ParametricPulse implements these property methods.

        # Need to use __dict__ property to get instance values inside the __getattr__.
        # Otherwise, run into the maximum recursion error. This class cannot define __slots__.
        defined_params = self.__dict__.get("_param_names", [])
        if item not in defined_params:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

        return self.__dict__["_param_vals"][defined_params.index(item)]

    @property
    def pulse_type(self) -> str:
        """Return display name of the pulse shape."""
        return self._pulse_type

    @property
    def envelope(self) -> "Expr":
        """Return envelope function of the pulse.

        See :ref:`symbolic_pulse_envelope` for details.

        A pulse instance without this value cannot generate waveform samples.
        Note that the expression should contain symbol ``t`` that represents a
        sampling time, along with parameters defined in :meth:`.parameters`.
        """
        return self._envelope_expr

    @envelope.setter
    def envelope(self, new_expr: "Expr"):
        """Add new envelope function to the pulse."""
        self._envelope_expr = new_expr

    @property
    def constraints(self) -> "Expr":
        """Returns pulse parameter constrains.

        See :ref:`symbolic_pulse_constraints` for details.

        A pulse instance without this value doesn't validate assigned parameters.
        """
        return self._consts_expr

    @constraints.setter
    def constraints(self, new_constraints: "Expr"):
        """Add new parameter constraints to the pulse."""
        self._consts_expr = new_constraints

    def get_waveform(self) -> Waveform:
        r"""Return a Waveform with samples filled according to the formula that the pulse
        represents and the parameter values it contains.

        Since the returned array is a discretized time series of the continuous function,
        this method uses a midpoint sampler. For ``duration``, return:

        .. math::

            \{f(t+0.5) \in \mathbb{C} | t \in \mathbb{Z} \wedge  0<=t<\texttt{duration}\}

        Returns:
            A waveform representation of this pulse.

        Raises:
            PulseError: When parameters are not bound.
        """
        if self.is_parameterized():
            raise PulseError("Unassigned parameter exists. All parameters must be assigned.")

        times = np.arange(0, self.duration) + 1 / 2
        args = (times, self.duration, self.amp, *self._param_vals)
        waveform = self._callable_envelope(*args)

        return Waveform(samples=waveform, name=self.name)

    def validate_parameters(self) -> None:
        """Validate parameters.

        Raises:
            PulseError: If the parameters passed are not valid.
        """
        if self.is_parameterized():
            return

        args = (self.duration, self.amp, *self._param_vals)
        if self._consts_expr is not None and not bool(self._callable_consts(*args)):
            param_repr = ", ".join(f"{p}={v}" for p, v in self.parameters.items())
            const_repr = str(self._consts_expr)
            raise PulseError(
                f"Assigned parameters {param_repr} violate following constraint: {const_repr}."
            )

        if self.limit_amplitude and _validate_amplitude_limit(self):
            # Check max amplitude limit by generating waveform.
            param_repr = ", ".join(f"{p}={v}" for p, v in self.parameters.items())
            raise PulseError(
                f"Maximum pulse amplitude norm exceeds 1.0 with assigned parameters {param_repr}."
                "This can be overruled by setting Pulse.limit_amplitude."
            )

    def is_parameterized(self) -> bool:
        """Return True iff the instruction is parameterized."""
        args = (self.duration, self.amp, *self._param_vals)
        return any(isinstance(val, ParameterExpression) for val in args)

    @property
    def parameters(self) -> Dict[str, Any]:
        params = {"duration": self.duration, "amp": self.amp}
        params.update(dict(zip(self._param_names, self._param_vals)))
        return params

    def __eq__(self, other: "SymbolicPulse") -> bool:
        # Not aware of expressions.
        if not isinstance(other, SymbolicPulse):
            return False
        if self._pulse_type != other._pulse_type:
            # Use pulse type equality rather than class.
            return False
        if self.parameters != other.parameters:
            return False
        return True

    def __hash__(self) -> int:
        return hash(
            (self._pulse_type, self.duration, self.amp, *self._param_names, *self._param_vals)
        )

    def __repr__(self) -> str:
        param_repr = ", ".join(f"{p}={v}" for p, v in self.parameters.items())
        return "{}({}{})".format(
            self._pulse_type,
            param_repr,
            f", name='{self.name}'" if self.name is not None else "",
        )


class Gaussian(SymbolicPulse):
    r"""A lifted and truncated pulse envelope shaped according to the Gaussian function whose
    mean is centered at the center of the pulse (duration / 2):

    .. math::

        f'(x) &= \exp\Bigl( -\frac12 \frac{{(x - \text{duration}/2)}^2}{\text{sigma}^2} \Bigr)\\
        f(x) &= \text{amp} \times \frac{f'(x) - f'(-1)}{1-f'(-1)}, \quad 0 \le x < \text{duration}

    where :math:`f'(x)` is the gaussian waveform without lifting or amplitude scaling.
    """

    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, ParameterExpression],
        sigma: Union[float, ParameterExpression],
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
    ):
        """Create new pulse instance.

        Args:
            duration: Pulse length in terms of the sampling period `dt`.
            amp: The amplitude of the Gaussian envelope.
            sigma: A measure of how wide or narrow the Gaussian peak is; described mathematically
                   in the class docstring.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                waveform to 1. The default is ``True`` and the amplitude is constrained to 1.

        """
        import sympy as sym

        parameters = {
            "duration": duration,
            "amp": amp,
            "sigma": sigma,
        }

        super().__init__(
            pulse_type=self.__class__.__name__,
            parameters=parameters,
            name=name,
            limit_amplitude=limit_amplitude,
        )

        # Add definitions
        t, duration, amp, sigma = sym.symbols("t, duration, amp, sigma")
        center = duration / 2

        self.envelope = amp * _lifted_gaussian(t, center, duration + 1, sigma)
        self.constraints = sigma > 0
        self.validate_parameters()


class GaussianSquare(SymbolicPulse):
    """A square pulse with a Gaussian shaped risefall on both sides lifted such that
    its first sample is zero.

    Exactly one of the ``risefall_sigma_ratio`` and ``width`` parameters has to be specified.

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
    """

    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, ParameterExpression],
        sigma: Union[float, ParameterExpression],
        width: Optional[Union[float, ParameterExpression]] = None,
        risefall_sigma_ratio: Optional[Union[float, ParameterExpression]] = None,
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
    ):
        """Create new pulse instance.

        Args:
            duration: Pulse length in terms of the sampling period `dt`.
            amp: The amplitude of the Gaussian and of the square pulse.
            sigma: A measure of how wide or narrow the Gaussian risefall is; see the class
                   docstring for more details.
            width: The duration of the embedded square pulse.
            risefall_sigma_ratio: The ratio of each risefall duration to sigma.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                waveform to 1. The default is ``True`` and the amplitude is constrained to 1.

        Raises:
            PulseError: When width and risefall_sigma_ratio are both empty or both non-empty.
        """
        import sympy as sym

        # Convert risefall_sigma_ratio into width which is defined in OpenPulse spec
        if width is None and risefall_sigma_ratio is None:
            raise PulseError(
                "Either the pulse width or the risefall_sigma_ratio parameter must be specified."
            )
        if width is not None and risefall_sigma_ratio is not None:
            raise PulseError(
                "Either the pulse width or the risefall_sigma_ratio parameter can be specified"
                " but not both."
            )
        if width is None and risefall_sigma_ratio is not None:
            width = duration - 2.0 * risefall_sigma_ratio * sigma

        parameters = {
            "duration": duration,
            "amp": amp,
            "sigma": sigma,
            "width": width,
        }

        super().__init__(
            pulse_type=self.__class__.__name__,
            parameters=parameters,
            name=name,
            limit_amplitude=limit_amplitude,
        )

        # Add definitions
        t, duration, amp, sigma, width = sym.symbols("t, duration, amp, sigma, width")
        center = duration / 2

        sq_t0 = center - width / 2
        sq_t1 = center + width / 2

        gaussian_ledge = _lifted_gaussian(t, sq_t0, -1, sigma)
        gaussian_redge = _lifted_gaussian(t, sq_t1, duration + 1, sigma)

        self.envelope = amp * sym.Piecewise(
            (gaussian_ledge, t <= sq_t0), (gaussian_redge, t >= sq_t1), (1, True)
        )
        self.constraints = sym.And(sigma > 0, width >= 0, duration >= width)
        self.validate_parameters()

    @property
    def risefall_sigma_ratio(self):
        """Return risefall_sigma_ratio. This is auxiliary parameter to define width."""
        return (self.duration - self.width) / (2.0 * self.sigma)


class Drag(SymbolicPulse):
    """The Derivative Removal by Adiabatic Gate (DRAG) pulse is a standard Gaussian pulse
    with an additional Gaussian derivative component and lifting applied.

    It is designed to reduce the frequency spectrum of a standard Gaussian pulse near
    the :math:`|1\\rangle\\leftrightarrow|2\\rangle` transition,
    reducing the chance of leakage to the :math:`|2\\rangle` state.

    .. math::

        g(x) &= \\exp\\Bigl(-\\frac12 \\frac{(x - \\text{duration}/2)^2}{\\text{sigma}^2}\\Bigr)\\\\
        g'(x) &= \\text{amp}\\times\\frac{g(x)-g(-1)}{1-g(-1)}\\\\
        f(x) &=  g'(x) \\times \\Bigl(1 + 1j \\times \\text{beta} \\times\
            \\Bigl(-\\frac{x - \\text{duration}/2}{\\text{sigma}^2}\\Bigr)  \\Bigr),
            \\quad 0 \\le x < \\text{duration}

    where :math:`g(x)` is a standard unlifted Gaussian waveform and
    :math:`g'(x)` is the lifted :class:`~qiskit.pulse.library.Gaussian` waveform.

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

    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, ParameterExpression],
        sigma: Union[float, ParameterExpression],
        beta: Union[float, ParameterExpression],
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
    ):
        """Create new pulse instance.

        Args:
            duration: Pulse length in terms of the sampling period `dt`.
            amp: The amplitude of the Drag envelope.
            sigma: A measure of how wide or narrow the Gaussian peak is; described mathematically
                   in the class docstring.
            beta: The correction amplitude.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                waveform to 1. The default is ``True`` and the amplitude is constrained to 1.
        """
        import sympy as sym

        parameters = {
            "duration": duration,
            "amp": amp,
            "sigma": sigma,
            "beta": beta,
        }

        super().__init__(
            pulse_type=self.__class__.__name__,
            parameters=parameters,
            name=name,
            limit_amplitude=limit_amplitude,
        )

        # Add definitions
        t, duration, amp, sigma, beta = sym.symbols("t, duration, amp, sigma, beta")
        center = duration / 2

        gauss = amp * _lifted_gaussian(t, center, duration + 1, sigma)
        deriv = -(t - center) / (sigma**2) * gauss

        self.envelope = gauss + 1j * beta * deriv
        self.constraints = sym.And(sigma > 0, sym.Eq(sym.im(beta), 0))
        self.validate_parameters()


class Constant(SymbolicPulse):
    """A simple constant pulse, with an amplitude value and a duration:

    .. math::

        f(x) = amp    ,  0 <= x < duration
        f(x) = 0      ,  elsewhere
    """

    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, ParameterExpression],
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
    ):
        """Create new pulse instance.

        Args:
            duration: Pulse length in terms of the sampling period `dt`.
            amp: The amplitude of the constant square pulse.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                waveform to 1. The default is ``True`` and the amplitude is constrained to 1.
        """
        import sympy as sym

        parameters = {
            "duration": duration,
            "amp": amp,
        }

        super().__init__(
            pulse_type=self.__class__.__name__,
            parameters=parameters,
            name=name,
            limit_amplitude=limit_amplitude,
        )

        # Add definitions
        t, duration, amp = sym.symbols("t, duration, amp")

        # Note this is implemented using Piecewise instead of just returning amp
        # directly because otherwise the expression has no t dependence and sympy's
        # lambdify will produce a function f that for an array t returns amp
        # instead of amp * np.ones(t.shape). This does not work well with
        # ParametricPulse.get_waveform().
        #
        # See: https://github.com/sympy/sympy/issues/5642
        self.envelope = amp * sym.Piecewise((1, (t >= 0) & (t <= duration)), (0, True))
        self.validate_parameters()
