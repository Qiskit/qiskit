# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Pulse envelope generation.
"""

import ast
import cmath
import inspect

import numpy as np
from matplotlib import get_backend as get_matplotlib_backend, pyplot as plt
from scipy.interpolate import CubicSpline

from qiskit.exceptions import QiskitError

MATH = ['exp', 'log', 'log10', 'sqrt', 'acos', 'asin', 'atan', 'cos', 'sin', 'tan',
        'acosh', 'asinh', 'atanh', 'cosh', 'sinh', 'tanh', 'pi', 'e', 'tau']


class Pulse:
    """Pulse specification."""

    def __init__(self, envfunc, t_gate, t_intv, **kwargs):
        """Create a new pulse envelope.
        Args:
            envfunc (str): equation describing pulse envelope
            t_gate (float): gate time
            t_intv (float): time interval of sampling points
            **kwargs: initial values of pulse parameter
        Raises:
            QiskitError: when pulse envelope is not in the correct format.
        """
        self._param = {}
        self._ast = None
        self.t_intv = t_intv
        self.t_gate = t_gate

        if isinstance(envfunc, str):
            self._ast, pars = Pulse.eval_expr(envfunc)
            self._param = {par: kwargs.get(par, 0) for par in pars}
        else:
            raise QiskitError('Incorrect specification of pulse envelope function.')

    @property
    def param(self):
        """Get parameters for describing pulse envelope

        Returns:
            dict: pulse parameters
        """
        return self._param

    @param.setter
    def param(self, param_new):
        """Set parameters for describing pulse envelope

        Args:
            param_new (dict): dictionary of parameters
        Raises:
            QiskitError: when pulse parameter is not in the correct format.
        """
        if isinstance(param_new, dict):
            for key, val in self._param.items():
                self._param[key] = param_new.get(key, val)
        else:
            raise QiskitError('Pulse parameter should be dictionary.')

    def tolist(self):
        """Calculate and output pulse envelope as a list of complex values [re, im]

        Returns:
            list: complex pulse envelope at each sampling point
        Raises:
            QiskitError: when envelope function is not initialized.
        """
        if self._ast:
            math_funcs = dict(inspect.getmembers(cmath))
            _dict = {k: math_funcs.get(k, None) for k in MATH}
            _dict.update(self._param)
            envelope = []

            _time = 0
            while _time <= self.t_gate:
                _dict['t'] = _time
                value = eval(compile(self._ast, filename='', mode='eval'),
                             {"__builtins__": None}, _dict)
                envelope.append([np.real(value), np.imag(value)])
                _time += self.t_intv
        else:
            raise QiskitError('Envelope function is not defined.')

        return envelope

    def plot(self, nop=1000):
        """Visualize pulse envelope.

        Args:
            nop (int): number of points for interpolation
        Returns:
            matplotlib.figure: a matplotlib figure object for the pulse envelope
        """
        pulse_samp = np.array(self.tolist())
        re_y = np.array(pulse_samp[:, 0])
        im_y = np.array(pulse_samp[:, 1])
        x = self.t_intv * np.linspace(0, len(re_y) - 1, len(re_y))

        # spline interpolation
        cs_ry = CubicSpline(x, re_y)
        cs_iy = CubicSpline(x, im_y)
        x_interp = np.linspace(0, max(x), nop)

        figure = plt.figure(figsize=(6, 5))
        ax0 = figure.add_subplot(111)
        ax0.scatter(x=x, y=re_y, c='red')
        ax0.scatter(x=x, y=im_y, c='blue')
        ax0.fill_between(x=x_interp, y1=cs_ry(x_interp), y2=np.zeros_like(x_interp),
                         facecolors='r', alpha=0.5)
        ax0.fill_between(x=x_interp, y1=cs_iy(x_interp), y2=np.zeros_like(x_interp),
                         facecolors='b', alpha=0.5)
        ax0.set_xlim([0, self.t_gate])
        ax0.grid(b=True, linestyle='-')

        if get_matplotlib_backend() == 'module://ipykernel.pylab.backend_inline':
            # returns None when matplotlib is inline mode to prevent Jupyter
            # with matplotlib inlining enabled to draw the diagram twice.
            img = None
        else:
            img = figure

        return img

    @staticmethod
    def eval_expr(expr_str):
        """Safety check of the input string

        Args:
            expr_str (str): equation string to be evaluated

        Returns:
            tuple[AST, list]: a tuple in the form (AST, list)
        Raises:
            QiskitError: when unintentional code is injected.
        """
        tree = ast.parse(expr_str, mode='eval')
        param = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id not in MATH and node.id not in param and node.id != 't':
                    param.append(node.id)
            elif isinstance(node, (ast.Expression, ast.Call, ast.Load,
                                   ast.BinOp, ast.UnaryOp, ast.operator,
                                   ast.unaryop, ast.cmpop, ast.Num)):
                continue
            else:
                raise QiskitError('Invalid function is used in the equation.')

        return tree, param
