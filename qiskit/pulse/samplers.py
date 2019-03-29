# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-return-doc

"""Sampler module for sampling of analytic pulses to discrete pulses.

Some atypical boilerplate has been added to solve the problem of decorators not preserving
their wrapped function signatures. Below we explain the problem that samplers solve and how
we implement this.

A sampler is a function that takes an analytic pulse function with signature:
    ```python
    def f(times: np.ndarray, *args, **kwargs) -> np.ndarray:
        ...
    ```
and returns a new function
    def f(duration: int, *args, **kwargs) -> SamplePulse:
        ...

Samplers are used to build up pulse commands from analytic pulse functions.

In Python the creation of a dynamic function that wraps another function will cause
the underlying signature and documentation of the underlying function to be overwritten.
In order to circumvent this issue the Python standard library provides the decorator
`functools.wraps` which allows the programmer to expose the names and signature of the
wrapped function as those of the dynamic function.

Samplers are implemented by creating a function with signature
    @sampler
    def left(analytic_pulse: Callable, duration: int, *args, **kwargs)
        ...

This will create a sampler function for `left`. Since it is a dynamic function it would not
have the docstring of `left` available too `help`. This could be fixed by wrapping with
`functools.wraps` in the `sampler`, but this would then cause the signature to be that of the
sampler function which is called on the analytic pulse, below:
    `(analytic_pulse: Callable, duration: int, *args, **kwargs)``
This is not correct for the sampler as the output sampled functions accept only a function.
For the standard sampler we get around this by not using `functools.wraps` and
explicitly defining our samplers such as `left`, `right` and `midpoint` and
calling `sampler` internally on the function that implements the sampling schemes such as
`_left`, `_right` and `_midpoint` respectively. See `left` for an example of this.

In this way our standard samplers will expose the proper help signature, but a user can
still create their own sampler with
    @sampler
    def custom_sampler(time, *args, **kwargs):
        ...
However, in this case it will be missing documentation of the underlying sampling methods.
We believe that the definition of custom samplers will be rather infrequent.

However, users will frequently apply sampler instances too analytic pulses. Therefore, a different
approach was required for sampled analytic functions (the output of an analytic pulse function
decorated by a sampler instance).

A sampler instance is a decorator that may be used to wrap analytic pulse functions such as
linear below:
```python
    @left
    def linear(times: np.ndarray, m: float, b: float) -> np.ndarray:
        ```Linear test function
        Args:
            times: Input times.
            m: Slope.
            b: Intercept
        Returns:
            np.ndarray
        ```
        return m*times+b
```
Which after decoration may be called with a duration rather than an array of times
    ```python
    duration = 10
    pulse_command = linear(10, 0.1, 0.1)
    ```
If one calls help on `linear` they will find
    ```
    linear(duration:int, *args, **kwargs) -> numpy.ndarray
    Discretized analytic pulse function: `linear` using
    sampler: `_left`.

     The first argument (time) of the analytic pulse function has been replaced with
     a discretized `duration` of type (int).

     Args:
         duration (int)
         *args: Remaining arguments of analytic pulse function.
                See analytic pulse function documentation below.
         **kwargs: Remaining kwargs of analytic pulse function.
                   See analytic pulse function documentation below.

     Sampled analytic function:

        function linear in module test.python.pulse.test_samplers
        linear(x:numpy.ndarray, m:float, b:float) -> numpy.ndarray
            Linear test function
            Args:
                x: Input times.
                m: Slope.
                b: Intercept
            Returns:
                np.ndarray
    ```
This is partly because `functools.wraps` has been used on the underlying function.
This in itself is not sufficient as the signature of the sampled function has
`duration`, whereas the signature of the analytic function is `time`.

This is acheived by removing `__wrapped__` set by `functools.wraps` in order to preserve
the correct signature and also applying `_update_annotations` and `_update_docstring`
to the generated function which corrects the function annotations and adds an informative
docstring respectively.

The user therefore has access to the correct sampled function docstring in its entirety, while
still seeing the signature for the analytic pulse function and all of its arguments.
"""

import functools
from typing import Callable
import textwrap
import pydoc

import numpy as np

import qiskit.pulse.commands as commands


def _update_annotations(discretized_pulse: Callable) -> Callable:
    """Update annotations of discretized analytic pulse function with duration.

    Args:
        discretized_pulse: Discretized decorated analytic pulse.
    """
    undecorated_annotations = list(discretized_pulse.__annotations__.items())
    decorated_annotations = undecorated_annotations[1:]
    decorated_annotations.insert(0, ('duration', int))
    discretized_pulse.__annotations__ = dict(decorated_annotations)
    return discretized_pulse


def _update_docstring(discretized_pulse: Callable, sampler_inst: Callable) -> Callable:
    """Update annotations of discretized analytic pulse function.

    Args:
        discretized_pulse: Discretized decorated analytic pulse.
        sampler_inst: Applied sampler.
    """
    wrapped_docstring = pydoc.render_doc(discretized_pulse, '%s')
    header, body = wrapped_docstring.split('\n', 1)
    body = textwrap.indent(body, '                    ')
    wrapped_docstring = header+body
    updated_ds = """
                Discretized analytic pulse function: `{analytic_name}` using
                sampler: `{sampler_name}`.

                 The first argument (time) of the analytic pulse function has been replaced with
                 a discretized `duration` of type (int).

                 Args:
                     duration (int)
                     *args: Remaining arguments of analytic pulse function.
                            See analytic pulse function documentation below.
                     **kwargs: Remaining kwargs of analytic pulse function.
                               See analytic pulse function documentation below.

                 Sampled analytic function:

                    {analytic_doc}
                """.format(analytic_name=discretized_pulse.__name__,
                           sampler_name=sampler_inst.__name__,
                           analytic_doc=wrapped_docstring)

    discretized_pulse.__doc__ = updated_ds
    return discretized_pulse


def sampler(sample_function: Callable) -> Callable:
    """Sampler decorator base method.

    Samplers are used for converting an analytic function to a discretized pulse.

    They operate on a function with the signature:
        `def f(times: np.ndarray, *args, **kwargs) -> np.ndarray`
    Where `times` is a numpy array of floats with length `n_times` and the output array
    is a complex numpy array with length `n_times`. The output of the decorator is an
    instance of `FunctionalPulse` with signature:
        `def g(duration: int, *args, **kwargs) -> SamplePulse`

    Note if your analytic pulse function outputs a `complex` scalar rather than a
    `np.array`, you should first vectorize it before applying a sampler.

    This class implements the sampler boilerplate for the sampler.

    Args:
        sample_function: A sampler function to be decorated.
    """

    def generate_sampler(analytic_pulse: Callable) -> Callable:
        """Return a decorated sampler function."""

        @functools.wraps(analytic_pulse)
        def call_sampler(duration: int, *args, **kwargs) -> commands.SamplePulse:
            """Replace the call to the analytic function with a call to the sampler applied
            to the anlytic pulse function."""
            sampled_pulse = sample_function(analytic_pulse, duration, *args, **kwargs)
            return np.asarray(sampled_pulse, dtype=np.complex)

        # Update type annotations for wrapped analytic function to be discrete
        call_sampler = _update_annotations(call_sampler)
        # Update docstring with that of the sampler and include sampled function documentation.
        call_sampler = _update_docstring(call_sampler, sample_function)
        # Unset wrapped to return base sampler signature
        # but still get rest of benefits of wraps
        # such as __name__, __qualname__
        call_sampler.__dict__.pop('__wrapped__')
        # wrap with functional pulse
        return commands.functional_pulse(call_sampler)

    return generate_sampler


def left(analytic_pulse: Callable) -> Callable:
    r"""Left sampling strategy decorator.

    See `pulse.samplers.sampler` for more information.

    For `duration`, return:
        $$\{f(t) \in \mathbb{C} | t \in \mathbb{Z} \wedge  0<=t<\texttt{duration}\}$$

    Args:
        analytic_pulse: To sample.
    """
    def _left(analytic_pulse: Callable, duration: int, *args, **kwargs) -> np.ndarray:
        """Sampling strategy for decorator.
        Args:
            analytic_pulse: Analytic pulse function to sample.
            duration: Duration to sample for.
            *args: Analytic pulse function args.
            *kwargs: Analytic pulse function kwargs.
        """
        times = np.arange(duration)
        return analytic_pulse(times, *args, **kwargs)

    return sampler(_left)(analytic_pulse)


def right(analytic_pulse: Callable) -> Callable:
    r"""Right sampling strategy decorator.

    See `pulse.samplers.sampler` for more information.

    For `duration`, return:
        $$\{f(t) \in \mathbb{C} | t \in \mathbb{Z} \wedge  0<t<=\texttt{duration}\}$$

    Args:
        analytic_pulse: To sample.
    """
    def _right(analytic_pulse: Callable, duration: int, *args, **kwargs) -> np.ndarray:
        """Sampling strategy for decorator.
        Args:
            analytic_pulse: Analytic pulse function to sample.
            duration: Duration to sample for.
            *args: Analytic pulse function args.
            *kwargs: Analytic pulse function kwargs.
        """
        times = np.arange(1, duration+1)
        return analytic_pulse(times, *args, **kwargs)

    return sampler(_right)(analytic_pulse)


def midpoint(analytic_pulse: Callable) -> Callable:
    r"""Midpoint sampling strategy decorator.

    See `pulse.samplers.sampler` for more information.

    For `duration`, return:
        $$\{f(t+0.5) \in \mathbb{C} | t \in \mathbb{Z} \wedge  0<=t<\texttt{duration}\}$$

    Args:
        analytic_pulse: To sample.
    """
    def _midpoint(analytic_pulse: Callable, duration: int, *args, **kwargs) -> np.ndarray:
        """Sampling strategy for decorator.
        Args:
            analytic_pulse: Analytic pulse function to sample.
            duration: Duration to sample for.
            *args: Analytic pulse function args.
            *kwargs: Analytic pulse function kwargs.
        """
        times = np.arange(1/2, duration + 1/2)
        return analytic_pulse(times, *args, **kwargs)

    return sampler(_midpoint)(analytic_pulse)
