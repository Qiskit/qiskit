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

These are pulses which are described by symbolic equations for their envelopes and for their
parameter constraints.
"""

import functools
import warnings
from typing import Any, Dict, List, Optional, Union, Callable

import numpy as np

from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.library.pulse import Pulse
from qiskit.pulse.library.waveform import Waveform
from qiskit.utils import optionals as _optional

if _optional.HAS_SYMENGINE:
    import symengine as sym
else:
    import sympy as sym


def _lifted_gaussian(
    t: sym.Symbol,
    center: Union[sym.Symbol, sym.Expr, complex],
    t_zero: Union[sym.Symbol, sym.Expr, complex],
    sigma: Union[sym.Symbol, sym.Expr, complex],
) -> sym.Expr:
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
    # Sympy automatically does expand.
    # This causes expression inconsistency after qpy round-trip serializing through sympy.
    # See issue for details: https://github.com/symengine/symengine.py/issues/409
    t_shifted = (t - center).expand()
    t_offset = (t_zero - center).expand()

    gauss = sym.exp(-((t_shifted / sigma) ** 2) / 2)
    offset = sym.exp(-((t_offset / sigma) ** 2) / 2)

    return (gauss - offset) / (1 - offset)


@functools.lru_cache(maxsize=None)
def _is_amplitude_valid(symbolic_pulse: "SymbolicPulse") -> bool:
    """A helper function to validate maximum amplitude limit.

    Result is cached for better performance.

    Args:
        symbolic_pulse: A pulse to validate.

    Returns:
        Return True if no sample point exceeds 1.0 in absolute value.
    """
    try:
        # Instantiation of Waveform does automatic amplitude validation.
        symbolic_pulse.get_waveform()
        return True
    except PulseError:
        return False


def _get_expression_args(expr: sym.Expr, params: Dict[str, float]) -> List[float]:
    """A helper function to get argument to evaluate expression.

    Args:
        expr: Symbolic expression to evaluate.
        params: Dictionary of parameter, which is a superset of expression arguments.

    Returns:
        Arguments passed to the lambdified expression.

    Raises:
        PulseError: When a free symbol value is not defined in the pulse instance parameters.
    """
    args = []
    for symbol in sorted(expr.free_symbols, key=lambda s: s.name):
        if symbol.name == "t":
            # 't' is a special parameter to represent time vector.
            # This should be place at first to broadcast other parameters
            # in symengine lambdify function.
            times = np.arange(0, params["duration"]) + 1 / 2
            args.insert(0, times)
            continue
        try:
            args.append(params[symbol.name])
        except KeyError as ex:
            raise PulseError(
                f"Pulse parameter '{symbol.name}' is not defined for this instance. "
                "Please check your waveform expression is correct."
            ) from ex
    return args


class LambdifiedExpression:
    """Descriptor to lambdify symbolic expression with cache.

    When a new symbolic expression is assigned for the first time, :class:`.LambdifiedExpression`
    will internally lambdify the expressions and store the resulting callbacks in its cache.
    The next time it encounters the same expression it will return the cached callbacks
    thereby increasing the code's speed.

    Note that this class is a python `Descriptor`_, and thus not intended to be
    directly called by end-users. This class is designed to be attached to the
    :class:`.SymbolicPulse` as attributes for symbolic expressions.

    _`Descriptor`: https://docs.python.org/3/reference/datamodel.html#descriptors
    """

    def __init__(self, attribute: str):
        """Create new descriptor.

        Args:
            attribute: Name of attribute of :class:`.SymbolicPulse` that returns
                the target expression to evaluate.
        """
        self.attribute = attribute
        self.lambda_funcs = dict()

    def __get__(self, instance, owner) -> Callable:
        expr = getattr(instance, self.attribute, None)
        if expr is None:
            raise PulseError(f"'{self.attribute}' of '{instance.pulse_type}' is not assigned.")
        key = hash(expr)
        if key not in self.lambda_funcs:
            self.__set__(instance, expr)

        return self.lambda_funcs[key]

    def __set__(self, instance, value):
        key = hash(value)
        if key not in self.lambda_funcs:
            params = []
            for p in sorted(value.free_symbols, key=lambda s: s.name):
                if p.name == "t":
                    # Argument "t" must be placed at first. This is a vector.
                    params.insert(0, p)
                    continue
                params.append(p)

            if _optional.HAS_SYMENGINE:
                try:
                    lamb = sym.lambdify(params, [value], real=False)

                    def _wrapped_lamb(*args):
                        if isinstance(args[0], np.ndarray):
                            # When the args[0] is a vector ("t"), tile other arguments args[1:]
                            # to prevent evaluation from looping over each element in t.
                            t = args[0]
                            args = np.hstack(
                                (
                                    t.reshape(t.size, 1),
                                    np.tile(args[1:], t.size).reshape(t.size, len(args) - 1),
                                )
                            )
                        return lamb(args)

                    func = _wrapped_lamb
                except RuntimeError:
                    # Currently symengine doesn't support complex_double version for
                    # several functions such as comparison operator and piecewise.
                    # If expression contains these function, it fall back to sympy lambdify.
                    # See https://github.com/symengine/symengine.py/issues/406 for details.
                    import sympy

                    func = sympy.lambdify(params, value)
            else:
                func = sym.lambdify(params, value)

            self.lambda_funcs[key] = func


class SymbolicPulse(Pulse):
    r"""The pulse representation model with parameters and symbolic expressions.

    A symbolic pulse instance can be defined with an envelope and parameter constraints.
    Envelope and parameter constraints should be provided as symbolic expressions.
    Rather than creating a subclass, different pulse shapes can be distinguished by
    the instance attributes :attr:`SymbolicPulse.envelope` and :attr:`SymbolicPulse.pulse_type`.

    The symbolic expressions must be defined either with SymPy_ or Symengine_.
    Usually Symengine-based expression is much more performant for instantiation
    of the :class:`SymbolicPulse`, however, it doesn't support every functions available in SymPy.
    You may need to choose proper library depending on how you define your pulses.
    Symengine works in the most envelopes and constraints, and thus it is recommended to use
    this library especially when your program contains a lot of pulses.
    Also note that Symengine has the limited platform support and may not be available
    for your local system. Symengine is a required dependency for Qiskit on platforms
    that support it will always be installed along with Qiskit on macOS ``x86_64`` and ``arm64``,
    and Linux ``x86_64``, ``aarch64``, and ``ppc64le``.
    For 64-bit Windows users they will need to manual install it.
    For 32-bit platforms such as ``i686`` and ``armv7`` Linux, and on Linux ``s390x``
    there are no pre-compiled packages available and to use symengine you'll need to
    compile it from source. If Symengine is not available in your environment SymPy will be used.

    .. _SymPy: https://www.sympy.org/en/index.html
    .. _Symengine: https://symengine.org

    .. _symbolic_pulse_envelope:

    .. rubric:: Envelope function

    The waveform at time :math:`t` is generated by the :meth:`get_waveform` according to

    .. math::

        F(t, \Theta) = \times F(t, {\rm duration}, \overline{\rm params})

    where :math:`\Theta` is the set of full pulse parameters in the :attr:`SymbolicPulse.parameters`
    dictionary which must include the :math:`\rm duration`.
    Note that the :math:`F` is an envelope of the waveform, and a programmer must provide this
    as a symbolic expression. :math:`\overline{\rm params}` can be arbitrary complex values
    as long as they pass :meth:`.validate_parameters` and your quantum backend can accept.
    The time :math:`t` and :math:`\rm duration` are in units of dt, i.e. sample time resolution,
    and this function is sampled with a discrete time vector in :math:`[0, {\rm duration}]`
    sampling the pulse envelope at every 0.5 dt (middle sampling strategy) when
    the :meth:`SymbolicPulse.get_waveform` method is called.
    The sample data is not generated until this method is called
    thus a symbolic pulse instance only stores parameter values and waveform shape,
    which greatly reduces memory footprint during the program generation.


    .. _symbolic_pulse_constraints:

    .. rubric:: Constraint functions

    Constraints on the parameters are defined with an instance attribute
    :attr:`SymbolicPulse.constraints` which can be provided through the constructor.
    The constraints value must be a symbolic expression, which is a
    function of parameters to be validated and must return a boolean value
    being ``True`` when parameters are valid.
    If there are multiple conditions to be evaluated, these conditions can be
    concatenated with logical expressions such as ``And`` and ``Or`` in SymPy or Symengine.
    The symbolic pulse instance can be played only when the constraint function returns ``True``.
    The constraint is evaluated when :meth:`.validate_parameters` is called.


    .. _symbolic_pulse_eval_condition:

    .. rubric:: Maximum amplitude validation

    When you play a pulse in a quantum backend, you might face the restriction on the power
    that your waveform generator can handle. Usually, the pulse amplitude is normalized
    by this maximum power, namely :math:`\max |F| \leq 1`. This condition is
    evaluated along with above constraints when you set ``limit_amplitude = True`` in the constructor.
    To evaluate maximum amplitude of the waveform, we need to call :meth:`get_waveform`.
    However, this introduces a significant overhead in the validation, and this cannot be ignored
    when you repeatedly instantiate symbolic pulse instances.
    :attr:`SymbolicPulse.valid_amp_conditions` provides a condition to skip this waveform validation,
    and the waveform is not generated as long as this condition returns ``True``,
    so that `healthy` symbolic pulses are created very quick.
    For example, for a simple pulse shape like ``amp * cos(f * t)``, we know that
    pulse amplitude is valid as long as ``amp`` remains less than magnitude 1.0.
    So ``abs(amp) <= 1`` could be passed as :attr:`SymbolicPulse.valid_amp_conditions` to skip
    doing a full waveform evaluation for amplitude validation.
    This expression is provided through the constructor. If this is not provided,
    the waveform is generated everytime when :meth:`.validate_parameters` is called.


    .. rubric:: Examples

    This is how a user can instantiate a symbolic pulse instance.
    In this example, we instantiate a custom `Sawtooth` envelope.

    .. jupyter-execute::

        from qiskit.pulse.library import SymbolicPulse

        my_pulse = SymbolicPulse(
            pulse_type="Sawtooth",
            duration=100,
            parameters={"amp": 0.1, "freq": 0.05},
            name="pulse1",
        )

    Note that :class:`SymbolicPulse` can be instantiated without providing
    the envelope and constraints. However, this instance cannot generate waveforms
    without knowing the envelope definition. Now you need to provide the envelope.

    .. jupyter-execute::

        import sympy

        t, amp, freq = sympy.symbols("t, amp, freq")
        envelope = 2 * amp * (freq * t - sympy.floor(1 / 2 + freq * t))

        my_pulse = SymbolicPulse(
            pulse_type="Sawtooth",
            duration=100,
            parameters={"amp": 0.1, "freq": 0.05},
            envelope=envelope,
            name="pulse1",
        )

        my_pulse.draw()

    Likewise, you can define :attr:`SymbolicPulse.constraints` for ``my_pulse``.
    After providing the envelope definition, you can generate the waveform data.
    Note that it would be convenient to define a factory function that automatically
    accomplishes this procedure.

    .. code-block:: python

        def Sawtooth(duration, amp, freq, name):
            t, amp, freq = sympy.symbols("t, amp, freq")

            instance = SymbolicPulse(
                pulse_type="Sawtooth",
                duration=duration,
                parameters={"amp": amp, "freq": freq},
                envelope=2 * amp * (freq * t - sympy.floor(1 / 2 + freq * t)),
                name=name,
            )

            return instance

    You can also provide a :class:`Parameter` object in the ``parameters`` dictionary,
    or define ``duration`` with a :class:`Parameter` object when you instantiate
    the symbolic pulse instance.
    A waveform cannot be generated until you assign all unbounded parameters.
    Note that parameters will be assigned through the schedule playing the pulse.


    .. _symbolic_pulse_serialize:

    .. rubric:: Serialization

    The :class:`~SymbolicPulse` subclass can be serialized along with the
    symbolic expressions through :mod:`qiskit.qpy`.
    A user can therefore create a custom pulse subclass with a novel envelope and constraints,
    and then one can instantiate the class with certain parameters to run on a backend.
    This pulse instance can be saved in the QPY binary, which can be loaded afterwards
    even within the environment not having original class definition loaded.
    This mechanism also allows us to easily share a pulse program including
    custom pulse instructions with collaborators.
    """

    __slots__ = (
        "_pulse_type",
        "_params",
        "_envelope",
        "_constraints",
        "_valid_amp_conditions",
    )

    # Lambdify caches keyed on sympy expressions. Returns the corresponding callable.
    _envelope_lam = LambdifiedExpression("_envelope")
    _constraints_lam = LambdifiedExpression("_constraints")
    _valid_amp_conditions_lam = LambdifiedExpression("_valid_amp_conditions")

    def __init__(
        self,
        pulse_type: str,
        duration: Union[ParameterExpression, int],
        parameters: Optional[Dict[str, Union[ParameterExpression, complex]]] = None,
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
        envelope: Optional[sym.Expr] = None,
        constraints: Optional[sym.Expr] = None,
        valid_amp_conditions: Optional[sym.Expr] = None,
    ):
        """Create a parametric pulse.

        Args:
            pulse_type: Display name of this pulse shape.
            duration: Duration of pulse.
            parameters: Dictionary of pulse parameters that defines the pulse envelope.
            name: Display name for this particular pulse envelope.
            limit_amplitude: If ``True``, then limit the absolute value of the amplitude of the
                waveform to 1. The default is ``True`` and the amplitude is constrained to 1.
            envelope: Pulse envelope expression.
            constraints: Pulse parameter constraint expression.
            valid_amp_conditions: Extra conditions to skip a full-waveform check for the
                amplitude limit. If this condition is not met, then the validation routine
                will investigate the full-waveform and raise an error when the amplitude norm
                of any data point exceeds 1.0. If not provided, the validation always
                creates a full-waveform.

        Raises:
            PulseError: When not all parameters are listed in the attribute :attr:`PARAM_DEF`.
        """
        super().__init__(
            duration=duration,
            name=name,
            limit_amplitude=limit_amplitude,
        )
        if parameters is None:
            parameters = {}

        self._pulse_type = pulse_type
        self._params = parameters

        self._envelope = envelope
        self._constraints = constraints
        self._valid_amp_conditions = valid_amp_conditions

    def __getattr__(self, item):
        # Get pulse parameters with attribute-like access.
        params = object.__getattribute__(self, "_params")
        if item not in params:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")
        return params[item]

    @property
    def pulse_type(self) -> str:
        """Return display name of the pulse shape."""
        return self._pulse_type

    @property
    def envelope(self) -> sym.Expr:
        """Return symbolic expression for the pulse envelope."""
        return self._envelope

    @property
    def constraints(self) -> sym.Expr:
        """Return symbolic expression for the pulse parameter constraints."""
        return self._constraints

    @property
    def valid_amp_conditions(self) -> sym.Expr:
        """Return symbolic expression for the pulse amplitude constraints."""
        return self._valid_amp_conditions

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
            PulseError: When parameters are not assigned.
            PulseError: When expression for pulse envelope is not assigned.
        """
        if self.is_parameterized():
            raise PulseError("Unassigned parameter exists. All parameters must be assigned.")

        if self._envelope is None:
            raise PulseError("Pulse envelope expression is not assigned.")

        fargs = _get_expression_args(self._envelope, self.parameters)
        return Waveform(samples=self._envelope_lam(*fargs), name=self.name)

    def validate_parameters(self) -> None:
        """Validate parameters.

        Raises:
            PulseError: If the parameters passed are not valid.
        """
        if self.is_parameterized():
            return

        if self._constraints is not None:
            fargs = _get_expression_args(self._constraints, self.parameters)
            if not bool(self._constraints_lam(*fargs)):
                param_repr = ", ".join(f"{p}={v}" for p, v in self.parameters.items())
                const_repr = str(self._constraints)
                raise PulseError(
                    f"Assigned parameters {param_repr} violate following constraint: {const_repr}."
                )

        if self._limit_amplitude:
            if self._valid_amp_conditions is not None:
                fargs = _get_expression_args(self._valid_amp_conditions, self.parameters)
                check_full_waveform = not bool(self._valid_amp_conditions_lam(*fargs))
            else:
                check_full_waveform = True

            if check_full_waveform:
                # Check full waveform only when the condition is satisified or
                # evaluation condition is not provided.
                # This operation is slower due to overhead of 'get_waveform'.
                if not _is_amplitude_valid(self):
                    param_repr = ", ".join(f"{p}={v}" for p, v in self.parameters.items())
                    raise PulseError(
                        f"Maximum pulse amplitude norm exceeds 1.0 with parameters {param_repr}."
                        "This can be overruled by setting Pulse.limit_amplitude."
                    )

    def is_parameterized(self) -> bool:
        """Return True iff the instruction is parameterized."""
        return any(isinstance(val, ParameterExpression) for val in self.parameters.values())

    @property
    def parameters(self) -> Dict[str, Any]:
        params = {"duration": self.duration}
        params.update(self._params)
        return params

    def __eq__(self, other: "SymbolicPulse") -> bool:

        if not isinstance(other, SymbolicPulse):
            return NotImplemented

        if self._pulse_type != other._pulse_type:
            return False

        if self._envelope != other._envelope:
            return False

        if self.parameters != other.parameters:
            return False

        return True

    def __hash__(self) -> int:
        if self.is_parameterized():
            raise NotImplementedError(
                "Hashing a symbolic pulse with unassigned parameter is not supported."
            )
        return hash((self._pulse_type, self._envelope, self.duration, *tuple(self._params.items())))

    def __repr__(self) -> str:
        param_repr = ", ".join(f"{p}={v}" for p, v in self.parameters.items())
        return "{}({}{})".format(
            self._pulse_type,
            param_repr,
            f", name='{self.name}'" if self.name is not None else "",
        )


class _PulseType(type):
    """Metaclass to warn at isinstance check."""

    def __instancecheck__(cls, instance):
        cls_alias = getattr(cls, "alias", None)

        # TODO promote this to Deprecation warning in future.
        #  Once type information usage is removed from user code,
        #  we will convert pulse classes into functions.
        warnings.warn(
            "Typechecking with the symbolic pulse subclass will be deprecated. "
            f"'{cls_alias}' subclass instance is turned into SymbolicPulse instance. "
            f"Use self.pulse_type == '{cls_alias}' instead.",
            PendingDeprecationWarning,
        )

        if not isinstance(instance, SymbolicPulse):
            return False
        return instance.pulse_type == cls_alias

    def __getattr__(cls, item):
        # For pylint. A SymbolicPulse subclass must implement several methods
        # such as .get_waveform and .validate_parameters.
        # In addition, they conventionally offer attribute-like access to the pulse parameters,
        # for example, instance.amp returns instance._params["amp"].
        # If pulse classes are directly instantiated, pylint yells no-member
        # since the pulse class itself implements nothing. These classes just
        # behave like a factory by internally instantiating the SymbolicPulse and return it.
        # It is not realistic to write disable=no-member across qiskit packages.
        return NotImplemented


class Gaussian(metaclass=_PulseType):
    r"""A lifted and truncated pulse envelope shaped according to the Gaussian function whose
    mean is centered at the center of the pulse (duration / 2):

    .. math::

        f'(x) &= \exp\Bigl( -\frac12 \frac{{(x - \text{duration}/2)}^2}{\text{sigma}^2} \Bigr)\\
        f(x) &= \text{A} \times  \frac{f'(x) - f'(-1)}{1-f'(-1)}, \quad 0 \le x < \text{duration}

    where :math:`f'(x)` is the gaussian waveform without lifting or amplitude scaling, and
    :math:`\text{A} = \text{amp} \times \exp\left(i\times\text{angle}\right)`.
    """

    alias = "Gaussian"

    def __new__(
        cls,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, float, ParameterExpression],
        sigma: Union[float, ParameterExpression],
        angle: Optional[Union[float, ParameterExpression]] = None,
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
    ) -> SymbolicPulse:
        """Create new pulse instance.

        Args:
            duration: Pulse length in terms of the sampling period `dt`.
            amp: The magnitude of the amplitude of the Gaussian envelope.
                    Complex amp support will be deprecated.
            sigma: A measure of how wide or narrow the Gaussian peak is; described mathematically
                   in the class docstring.
            angle: The angle of the complex amplitude of the Gaussian envelope. Default value 0.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                waveform to 1. The default is ``True`` and the amplitude is constrained to 1.

        Returns:
            SymbolicPulse instance.

        Raises:
            PulseError: If both complex amp and angle are provided as arguments.
        """
        # This should be removed once complex amp support is deprecated.
        if isinstance(amp, complex):
            if angle is None:
                warnings.warn(
                    "Complex amp will be deprecated. "
                    "Use float amp (for the magnitude) and float angle instead.",
                    PendingDeprecationWarning,
                )
            else:
                raise PulseError("amp can't be complex when providing angle")

        if angle is None:
            angle = 0

        parameters = {"amp": amp, "sigma": sigma, "angle": angle}

        # Prepare symbolic expressions
        _t, _duration, _amp, _sigma, _angle = sym.symbols("t, duration, amp, sigma, angle")
        _center = _duration / 2

        envelope_expr = (
            _amp * sym.exp(sym.I * _angle) * _lifted_gaussian(_t, _center, _duration + 1, _sigma)
        )

        consts_expr = _sigma > 0
        valid_amp_conditions_expr = sym.Abs(_amp) <= 1.0

        instance = SymbolicPulse(
            pulse_type=cls.alias,
            duration=duration,
            parameters=parameters,
            name=name,
            limit_amplitude=limit_amplitude,
            envelope=envelope_expr,
            constraints=consts_expr,
            valid_amp_conditions=valid_amp_conditions_expr,
        )
        instance.validate_parameters()

        return instance


class GaussianSquare(metaclass=_PulseType):
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
        f(x) &= \\text{A} \\times \\frac{f'(x) - f'(-1)}{1-f'(-1)},\
            \\quad 0 \\le x < \\text{duration}

    where :math:`f'(x)` is the gaussian square waveform without lifting or amplitude scaling, and
    :math:`\\text{A} = \\text{amp} \\times \\exp\\left(i\\times\\text{angle}\\right)`.
    """

    alias = "GaussianSquare"

    def __new__(
        cls,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, float, ParameterExpression],
        sigma: Union[float, ParameterExpression],
        width: Optional[Union[float, ParameterExpression]] = None,
        angle: Optional[Union[float, ParameterExpression]] = None,
        risefall_sigma_ratio: Optional[Union[float, ParameterExpression]] = None,
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
    ) -> SymbolicPulse:
        """Create new pulse instance.

        Args:
            duration: Pulse length in terms of the sampling period `dt`.
            amp: The magnitude of the amplitude of the Gaussian and square pulse.
                    Complex amp support will be deprecated.
            sigma: A measure of how wide or narrow the Gaussian risefall is; see the class
                   docstring for more details.
            width: The duration of the embedded square pulse.
            angle: The angle of the complex amplitude of the pulse. Default value 0.
            risefall_sigma_ratio: The ratio of each risefall duration to sigma.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                waveform to 1. The default is ``True`` and the amplitude is constrained to 1.

        Returns:
            SymbolicPulse instance.

        Raises:
            PulseError: When width and risefall_sigma_ratio are both empty or both non-empty.
            PulseError: If both complex amp and angle are provided as arguments.
        """
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

        # This should be removed once complex amp support is deprecated.
        if isinstance(amp, complex):
            if angle is None:
                warnings.warn(
                    "Complex amp will be deprecated. "
                    "Use float amp (for the magnitude) and float angle instead.",
                    PendingDeprecationWarning,
                )
            else:
                raise PulseError("amp can't be complex when providing angle")

        if angle is None:
            angle = 0

        parameters = {"amp": amp, "sigma": sigma, "width": width, "angle": angle}

        # Prepare symbolic expressions
        _t, _duration, _amp, _sigma, _width, _angle = sym.symbols(
            "t, duration, amp, sigma, width, angle"
        )
        _center = _duration / 2

        _sq_t0 = _center - _width / 2
        _sq_t1 = _center + _width / 2

        _gaussian_ledge = _lifted_gaussian(_t, _sq_t0, -1, _sigma)
        _gaussian_redge = _lifted_gaussian(_t, _sq_t1, _duration + 1, _sigma)

        envelope_expr = (
            _amp
            * sym.exp(sym.I * _angle)
            * sym.Piecewise(
                (_gaussian_ledge, _t <= _sq_t0), (_gaussian_redge, _t >= _sq_t1), (1, True)
            )
        )

        consts_expr = sym.And(_sigma > 0, _width >= 0, _duration >= _width)
        valid_amp_conditions_expr = sym.Abs(_amp) <= 1.0

        instance = SymbolicPulse(
            pulse_type=cls.alias,
            duration=duration,
            parameters=parameters,
            name=name,
            limit_amplitude=limit_amplitude,
            envelope=envelope_expr,
            constraints=consts_expr,
            valid_amp_conditions=valid_amp_conditions_expr,
        )
        instance.validate_parameters()

        return instance


class Drag(metaclass=_PulseType):
    """The Derivative Removal by Adiabatic Gate (DRAG) pulse is a standard Gaussian pulse
    with an additional Gaussian derivative component and lifting applied.

    It can be calibrated either to reduce the phase error due to virtual population of the
    :math:`|2\\rangle` state during the pulse or to reduce the frequency spectrum of a
    standard Gaussian pulse near the :math:`|1\\rangle\\leftrightarrow|2\\rangle` transition,
    reducing the chance of leakage to the :math:`|2\\rangle` state.

    .. math::

        g(x) &= \\exp\\Bigl(-\\frac12 \\frac{(x - \\text{duration}/2)^2}{\\text{sigma}^2}\\Bigr)\\\\
        g'(x) &= \\text{A}\\times\\frac{g(x)-g(-1)}{1-g(-1)}\\\\
        f(x) &=  g'(x) \\times \\Bigl(1 + 1j \\times \\text{beta} \\times\
            \\Bigl(-\\frac{x - \\text{duration}/2}{\\text{sigma}^2}\\Bigr)  \\Bigr),
            \\quad 0 \\le x < \\text{duration}

    where :math:`g(x)` is a standard unlifted Gaussian waveform, :math:`g'(x)` is the lifted
    :class:`~qiskit.pulse.library.Gaussian` waveform, and
    :math:`\\text{A} = \\text{amp} \\times \\exp\\left(i\\times\\text{angle}\\right)`.

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

    alias = "Drag"

    def __new__(
        cls,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, float, ParameterExpression],
        sigma: Union[float, ParameterExpression],
        beta: Union[float, ParameterExpression],
        angle: Optional[Union[float, ParameterExpression]] = None,
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
    ) -> SymbolicPulse:
        """Create new pulse instance.

        Args:
            duration: Pulse length in terms of the sampling period `dt`.
            amp: The magnitude of the amplitude of the DRAG envelope.
                    Complex amp support will be deprecated.
            sigma: A measure of how wide or narrow the Gaussian peak is; described mathematically
                   in the class docstring.
            beta: The correction amplitude.
            angle: The angle of the complex amplitude of the DRAG envelope. Default value 0.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                waveform to 1. The default is ``True`` and the amplitude is constrained to 1.

        Returns:
            SymbolicPulse instance.

        Raises:
            PulseError: If both complex amp and angle are provided as arguments.
        """
        # This should be removed once complex amp support is deprecated.
        if isinstance(amp, complex):
            if angle is None:
                warnings.warn(
                    "Complex amp will be deprecated. "
                    "Use float amp (for the magnitude) and float angle instead.",
                    PendingDeprecationWarning,
                )
            else:
                raise PulseError("amp can't be complex when providing angle")

        if angle is None:
            angle = 0

        parameters = {"amp": amp, "sigma": sigma, "beta": beta, "angle": angle}

        # Prepare symbolic expressions
        _t, _duration, _amp, _sigma, _beta, _angle = sym.symbols(
            "t, duration, amp, sigma, beta, angle"
        )
        _center = _duration / 2

        _gauss = _lifted_gaussian(_t, _center, _duration + 1, _sigma)
        _deriv = -(_t - _center) / (_sigma**2) * _gauss

        envelope_expr = _amp * sym.exp(sym.I * _angle) * (_gauss + sym.I * _beta * _deriv)

        consts_expr = _sigma > 0
        valid_amp_conditions_expr = sym.And(sym.Abs(_amp) <= 1.0, sym.Abs(_beta) < _sigma)

        instance = SymbolicPulse(
            pulse_type="Drag",
            duration=duration,
            parameters=parameters,
            name=name,
            limit_amplitude=limit_amplitude,
            envelope=envelope_expr,
            constraints=consts_expr,
            valid_amp_conditions=valid_amp_conditions_expr,
        )
        instance.validate_parameters()

        return instance


class Constant(metaclass=_PulseType):
    """A simple constant pulse, with an amplitude value and a duration:

    .. math::

        f(x) = \\text{amp}\\times\\exp\\left(i\\text{angle}\\right)    ,  0 <= x < duration
        f(x) = 0      ,  elsewhere
    """

    alias = "Constant"

    def __new__(
        cls,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, float, ParameterExpression],
        angle: Optional[Union[float, ParameterExpression]] = None,
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
    ) -> SymbolicPulse:
        """Create new pulse instance.

        Args:
            duration: Pulse length in terms of the sampling period `dt`.
            amp: The magnitude of the amplitude of the square envelope.
                    Complex amp support will be deprecated.
            angle: The angle of the complex amplitude of the square envelope. Default value 0.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                waveform to 1. The default is ``True`` and the amplitude is constrained to 1.

        Returns:
            SymbolicPulse instance.

        Raises:
            PulseError: If both complex amp and angle are provided as arguments.
        """
        # This should be removed once complex amp support is deprecated.
        if isinstance(amp, complex):
            if angle is None:
                warnings.warn(
                    "Complex amp will be deprecated. "
                    "Use float amp (for the magnitude) and float angle instead.",
                    PendingDeprecationWarning,
                )
            else:
                raise PulseError("amp can't be complex when providing angle")

        if angle is None:
            angle = 0

        parameters = {"amp": amp, "angle": angle}

        # Prepare symbolic expressions
        _t, _amp, _duration, _angle = sym.symbols("t, amp, duration, angle")

        # Note this is implemented using Piecewise instead of just returning amp
        # directly because otherwise the expression has no t dependence and sympy's
        # lambdify will produce a function f that for an array t returns amp
        # instead of amp * np.ones(t.shape). This does not work well with
        # ParametricPulse.get_waveform().
        #
        # See: https://github.com/sympy/sympy/issues/5642
        envelope_expr = (
            _amp
            * sym.exp(sym.I * _angle)
            * sym.Piecewise((1, sym.And(_t >= 0, _t <= _duration)), (0, True))
        )

        valid_amp_conditions_expr = sym.Abs(_amp) <= 1.0

        instance = SymbolicPulse(
            pulse_type="Constant",
            duration=duration,
            parameters=parameters,
            name=name,
            limit_amplitude=limit_amplitude,
            envelope=envelope_expr,
            valid_amp_conditions=valid_amp_conditions_expr,
        )
        instance.validate_parameters()

        return instance
