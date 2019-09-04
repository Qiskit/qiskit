# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Optimizer interface"""

from enum import IntEnum
import logging
from abc import abstractmethod
import numpy as np

from qiskit.aqua import Pluggable

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class Optimizer(Pluggable):
    """Base class for optimization algorithm."""

    class SupportLevel(IntEnum):
        """ Support Level enum """
        not_supported = 0  # Does not support the corresponding parameter in optimize()
        ignored = 1        # Feature can be passed as non None but will be ignored
        supported = 2      # Feature is supported
        required = 3       # Feature is required and must be given, None is invalid

    DEFAULT_CONFIGURATION = {
        'support_level': {
            'gradient': SupportLevel.not_supported,
            'bounds': SupportLevel.not_supported,
            'initial_point': SupportLevel.not_supported
        },
        'options': []
    }

    @abstractmethod
    def __init__(self):
        """Constructor.

        Initialize the optimization algorithm, setting the support
        level for _gradient_support_level, _bound_support_level,
        _initial_point_support_level, and empty options.

        """
        super().__init__()
        if 'support_level' not in self._configuration:
            self._configuration['support_level'] = self.DEFAULT_CONFIGURATION['support_level']
        if 'options' not in self._configuration:
            self._configuration['options'] = self.DEFAULT_CONFIGURATION['options']
        self._gradient_support_level = self._configuration['support_level']['gradient']
        self._bounds_support_level = self._configuration['support_level']['bounds']
        self._initial_point_support_level = self._configuration['support_level']['initial_point']
        self._options = {}
        self._max_evals_grouped = 1

    @classmethod
    def init_params(cls, params):
        """Initialize with a params dictionary.

        A dictionary of config params as per the configuration object. Some of these params get
        passed to scipy optimizers in an options dictionary. We can specify an options array of
        names in config dictionary to have the options dictionary automatically populated. All
        other config items, excluding name, will be passed to init_args

        Args:
            params (dict): configuration dict
        Returns:
            Optimizer: instance of this class
        """
        opt_params = params.get(Pluggable.SECTION_KEY_OPTIMIZER)
        logger.debug('init_params: %s', opt_params)
        args = {k: v for k, v in opt_params.items() if k != 'name'}
        optimizer = cls(**args)
        return optimizer

    def set_options(self, **kwargs):
        """
        Sets or updates values in the options dictionary.

        The options dictionary may be used internally by a given optimizer to
        pass additional optional values for the underlying optimizer/optimization
        function used. The options dictionary may be initially populated with
        a set of key/values when the given optimizer is constructed.

        Args:
            kwargs (dict): options, given as name=value.
        """
        for name, value in kwargs.items():
            self._options[name] = value
        logger.debug('options: %s', self._options)

    @staticmethod
    def gradient_num_diff(x_center, f, epsilon, max_evals_grouped=1):
        """
        We compute the gradient with the numeric differentiation in the parallel way,
        around the point x_center.
        Args:
            x_center (ndarray): point around which we compute the gradient
            f (func): the function of which the gradient is to be computed.
            epsilon (float): the epsilon used in the numeric differentiation.
            max_evals_grouped (int): max evals grouped
        Returns:
            grad: the gradient computed

        """
        forig = f(*((x_center,)))
        grad = []
        ei = np.zeros((len(x_center),), float)
        todos = []
        for k in range(len(x_center)):
            ei[k] = 1.0
            d = epsilon * ei
            todos.append(x_center + d)
            ei[k] = 0.0

        counter = 0
        chunk = []
        chunks = []
        length = len(todos)
        # split all points to chunks, where each chunk has batch_size points
        for i in range(length):
            x = todos[i]
            chunk.append(x)
            counter += 1
            # the last one does not have to reach batch_size
            if counter == max_evals_grouped or i == length-1:
                chunks.append(chunk)
                chunk = []
                counter = 0

        for chunk in chunks:  # eval the chunks in order
            parallel_parameters = np.concatenate(chunk)
            todos_results = f(parallel_parameters)  # eval the points in a chunk (order preserved)
            if isinstance(todos_results, float):
                grad.append((todos_results - forig) / epsilon)
            else:
                for todor in todos_results:
                    grad.append((todor - forig) / epsilon)

        return np.array(grad)

    @staticmethod
    def wrap_function(function, args):
        """
        Wrap the function to implicitly inject the args at the call of the function.
        Args:
            function (func): the target function
            args (tuple): the args to be injected
        Returns:
            function_wrapper: wrapper
        """
        def function_wrapper(*wrapper_args):
            return function(*(wrapper_args + args))
        return function_wrapper

    @property
    def setting(self):
        """ return setting """
        ret = "Optimizer: {}\n".format(self._configuration['name'])
        params = ""
        for key, value in self.__dict__.items():
            if key != "_configuration" and key[0] == "_":
                params += "-- {}: {}\n".format(key[1:], value)
        ret += "{}".format(params)
        return ret

    @abstractmethod
    def optimize(self, num_vars, objective_function, gradient_function=None,
                 variable_bounds=None, initial_point=None):
        """Perform optimization.

        Args:
            num_vars (int) : number of parameters to be optimized.
            objective_function (callable) : handle to a function that
                computes the objective function.
            gradient_function (callable) : handle to a function that
                computes the gradient of the objective function, or
                None if not available.
            variable_bounds (list[(float, float)]) : list of variable
                bounds, given as pairs (lower, upper). None means
                unbounded.
            initial_point (numpy.ndarray[float]) : initial point.

        Returns:
            point, value, nfev
               point: is a 1D numpy.ndarray[float] containing the solution
               value: is a float with the objective function value
               nfev: number of objective function calls made if available or None
        Raises:
            ValueError: invalid input
        """

        if initial_point is not None and len(initial_point) != num_vars:
            raise ValueError('Initial point does not match dimension')
        if variable_bounds is not None and len(variable_bounds) != num_vars:
            raise ValueError('Variable bounds not match dimension')

        has_bounds = False
        if variable_bounds is not None:
            # If *any* value is *equal* in bounds array to None then the does *not* have bounds
            has_bounds = not np.any(np.equal(variable_bounds, None))

        if gradient_function is None and self.is_gradient_required:
            raise ValueError('Gradient is required but None given')
        if not has_bounds and self.is_bounds_required:
            raise ValueError('Variable bounds is required but None given')
        if initial_point is None and self.is_initial_point_required:
            raise ValueError('Initial point is required but None given')

        if gradient_function is not None and self.is_gradient_ignored:
            logger.debug(
                'WARNING: %s does not support gradient function. It will be ignored.',
                self.configuration['name'])
        if has_bounds and self.is_bounds_ignored:
            logger.debug(
                'WARNING: %s does not support bounds. It will be ignored.',
                self.configuration['name'])
        if initial_point is not None and self.is_initial_point_ignored:
            logger.debug(
                'WARNING: %s does not support initial point. It will be ignored.',
                self.configuration['name'])
        pass

    @property
    def gradient_support_level(self):
        """ returns gradient support level """
        return self._gradient_support_level

    @property
    def is_gradient_ignored(self):
        """ returns is gradient ignored """
        return self._gradient_support_level == self.SupportLevel.ignored

    @property
    def is_gradient_supported(self):
        """ returns is gradient supported """
        return self._gradient_support_level != self.SupportLevel.not_supported

    @property
    def is_gradient_required(self):
        """ returns is gradient required """
        return self._gradient_support_level == self.SupportLevel.required

    @property
    def bounds_support_level(self):
        """ returns bounds support level """
        return self._bounds_support_level

    @property
    def is_bounds_ignored(self):
        """ returns is bounds ignored """
        return self._bounds_support_level == self.SupportLevel.ignored

    @property
    def is_bounds_supported(self):
        """ returns is bounds supported """
        return self._bounds_support_level != self.SupportLevel.not_supported

    @property
    def is_bounds_required(self):
        """ returns is bounds required """
        return self._bounds_support_level == self.SupportLevel.required

    @property
    def initial_point_support_level(self):
        """ returns initial point support level """
        return self._initial_point_support_level

    @property
    def is_initial_point_ignored(self):
        """ return is initial point ignored """
        return self._initial_point_support_level == self.SupportLevel.ignored

    @property
    def is_initial_point_supported(self):
        """ returns is initial point supported """
        return self._initial_point_support_level != self.SupportLevel.not_supported

    @property
    def is_initial_point_required(self):
        """ returns is initial point required """
        return self._initial_point_support_level == self.SupportLevel.required

    def print_options(self):
        """Print algorithm-specific options."""
        for name in sorted(self._options):
            logger.debug('{:s} = {:s}'.format(name, str(self._options[name])))

    def set_max_evals_grouped(self, limit):
        """ set max evals grouped """
        self._max_evals_grouped = limit
