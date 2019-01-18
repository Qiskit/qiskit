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
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import get_backend as get_matplotlib_backend
from scipy.interpolate import CubicSpline

from qiskit.exceptions import QiskitError

builtin = ['exp', 'log', 'log10', 'sqrt', 'acos', 'asin', 'atan', 'cos', 'sin', 'tan',
           'acosh', 'asinh', 'atanh', 'cosh', 'sinh', 'tanh', 'pi', 'e', 'tau']


class Pulse(object):
    """Pulse specification."""

    def __init__(self, envfunc, t_gate, dt, **kwargs):
        """Create a new pulse envelope.
        Args:
            envfunc (str): equation describing pulse envelope
            t_gate (float): gate time
            dt (float): time resolution
        Raises:
            QiskitError: when pulse envelope is not in the correct format.
        """
        self._param = {}
        self._ast = None
        self.dt = dt
        self.t_gate = t_gate

        if isinstance(envfunc, str):
            self._ast, pars = Pulse.eval_expr(envfunc)
            self._param = dict([(par, kwargs.get(par, 0)) for par in pars])
        else:
            raise QiskitError('Incorrect specification of pulse envelope function.')

    @property
    def param(self):
        """Get parameters for describing pulse envelope

        Returns:
            param (dict): Dictionary of parameters
        """
        return self._param

    @param.setter
    def param(self, param_new):
        """Set parameters for describing pulse envelope

        Args:
            param_new (dict): Dictionary of parameters
        """
        if isinstance(param_new, dict):
            for key, val in self._param.items():
                self._param[key] = param_new.get(key, val)
        else:
            raise QiskitError('Pulse parameter should be dictionary.')

    def tolist(self):
        """Calculate and output pulse envelope as a list of complex values [re, im]

        Returns:
            envelope (list): List of pulse envelope at each time interval dt
        """
        if self._ast:
            math_funcs = dict(inspect.getmembers(cmath))
            _dict = dict([(k, math_funcs.get(k, None)) for k in builtin])
            _dict.update(self._param)
            envelope = []

            _t = 0
            while _t <= self.t_gate:
                _dict['t'] = _t
                value = eval(compile(self._ast, filename='', mode='eval'),
                             {"__builtins__": None}, _dict)
                envelope.append([np.real(value), np.imag(value)])
                _t += self.dt
        else:
            raise QiskitError('Envelope function is not defined.')

        return envelope

    def plot(self, nop=1000):
        """Visualize pulse envelope.

        Args:
            nop (int): number of points for interpolation
        Returns:
            im (matplotlib.figure): a matplotlib figure object for the pulse envelope
        """
        _v = np.array(self.tolist())
        ry = np.array(_v[:, 0])
        iy = np.array(_v[:, 1])
        x = self.dt * np.linspace(0, len(ry) - 1, len(ry))

        # spline interpolation
        cs_ry = CubicSpline(x, ry)
        cs_iy = CubicSpline(x, iy)
        xs = np.linspace(0, max(x), nop)

        figure = plt.figure(figsize=(6, 5))
        ax = figure.add_subplot(111)
        ax.scatter(x=x, y=ry, c='red')
        ax.scatter(x=x, y=iy, c='blue')
        ax.fill_between(x=xs, y1=cs_ry(xs), y2=np.zeros_like(xs), facecolors='r', alpha=0.5)
        ax.fill_between(x=xs, y1=cs_iy(xs), y2=np.zeros_like(xs), facecolors='b', alpha=0.5)
        ax.set_xlim([0, self.t_gate])
        ax.grid(b=True, linestyle='-')

        if get_matplotlib_backend() == 'module://ipykernel.pylab.backend_inline':
            # returns None when matplotlib is inline mode to prevent Jupyter
            # with matplotlib inlining enabled to draw the diagram twice.
            im = None
        else:
            im = figure

        return im

    @staticmethod
    def eval_expr(expr_str):
        """Safety check of the input string

        Args:
            expr_str (str): equation string to be evaluated

        Returns:
            tree (AST): abstract syntax tree of input equation string
            param (list): list of parameters
        """
        tree = ast.parse(expr_str, mode='eval')
        param = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id not in builtin and node.id not in param and node.id != 't':
                    param.append(node.id)
            elif isinstance(node, (ast.Expression, ast.Call, ast.Load,
                                   ast.BinOp, ast.UnaryOp, ast.operator,
                                   ast.unaryop, ast.cmpop, ast.Num)):
                continue
            else:
                raise QiskitError('Invalid function is used in the equation.')

        return tree, param
