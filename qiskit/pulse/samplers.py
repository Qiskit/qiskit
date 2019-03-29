# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-return-doc

"""Sampler module for sampling of analytic pulses to discrete pulses."""

import functools
from typing import Callable

import numpy as np

from .commands import FunctionalPulse


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


def _update_docstring(discretized_pulse: Callable, sampler: Callable) -> Callable:
    """Update annotations of discretized analytic pulse function.

    Args:
        discretized_pulse: Discretized decorated analytic pulse.
        sampler: Applied sampler.
    """

    updated_ds = """
     Discretized analytic pulse function: `{analytic_name}` using sampler: `{sampler_name}`.

     The first argument (time) of the analytic pulse function has been replaced with
     a discretized `duration` of type (int).

     Args:
         duration (int)
         *args: Remaining arguments of analytic pulse function.
                See analytic pulse function signature below.
         **kwargs: Remaining kwargs of analytic pulse function.
                   See analytic pulse function signature below.

    Analytic function docstring:

    {analytic_doc}
    """.format(analytic_name=discretized_pulse.__qualname__,
               sampler_name=sampler.__qualname__,
               analytic_doc=discretized_pulse.__doc__)

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

    @functools.wraps(sample_function)
    def generate_sampler(analytic_pulse: Callable) -> Callable:
        """Return a decorated sampler function."""

        @functools.wraps(analytic_pulse, updated=tuple())
        def call_sampler(duration: int, *args, **kwargs) -> FunctionalPulse:
            """Replace the call to the analytic function with a call to the sampler applied
            to the anlytic pulse function."""
            sampled_pulse = sample_function(analytic_pulse, duration, *args, **kwargs)
            return np.asarray(sampled_pulse, dtype=np.complex)

        # update type annotations for wrapped analytic function to be discrete
        call_sampler = _update_annotations(call_sampler)
        call_sampler = _update_docstring(call_sampler, sample_function)
        call_sampler.__dict__.pop('__wrapped__')
        # wrap with functional pulse
        return FunctionalPulse(call_sampler)

    return generate_sampler


@sampler
def left(analytic_pulse: Callable, duration: int, *args, **kwargs) -> np.ndarray:
    r"""Left sampling strategy decorator.

    See `pulse.samplers.sampler` for more information.

    For `duration`, return:
        $$\{f(t) \in \mathbb{C} | t \in \mathbb{Z} \wedge  0<=t<\texttt{duration}\}$$

    Args:
        analytic_pulse: Analytic pulse function to sample.
        duration: Duration to sample for.
        *args: Analytic pulse function args.
        *kwargs: Analytic pulse function kwargs.
    """
    times = np.arange(duration)
    return analytic_pulse(times, *args, **kwargs)


@sampler
def right(analytic_pulse: Callable, duration: int, *args, **kwargs) -> np.ndarray:
    r"""Right sampling strategy decorator.

    For `duration`, return:
        $$\{f(t) \in \mathbb{C} | t \in \mathbb{Z} \wedge  0<t<=\texttt{duration}\}$$

    Args:
        analytic_pulse: Analytic pulse function to sample.
        duration: Duration to sample for.
        *args: Analytic pulse function args.
        **kwargs: Analytic pulse function kwargs.
    """
    times = np.arange(1, duration+1)
    return analytic_pulse(times, *args, **kwargs)


@sampler
def midpoint(analytic_pulse: Callable, duration: int, *args, **kwargs) -> np.ndarray:
    r"""Midpoint sampling strategy decorator.

    For `duration`, return:
        $$\{f(t+0.5) \in \mathbb{C} | t \in \mathbb{Z} \wedge  0<=t<\texttt{duration}\}$$

    Args:
        analytic_pulse: Analytic pulse function to sample.
        duration: Duration to sample for.
        *args: Analytic pulse function args.
        **kwargs: Analytic pulse function kwargs.
    """
    times = np.arange(1/2, (duration) + 1/2)
    return analytic_pulse(times, *args, **kwargs)
