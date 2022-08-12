# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Tools to create LaTeX arrays.
"""

import math
from fractions import Fraction
import numpy as np

from qiskit.exceptions import MissingOptionalLibraryError

def _round_if_close(data):
    """Round real and imaginary parts of complex number of close to zero"""
    data = np.real_if_close(data)
    data = -1j * np.real_if_close(data * 1j)
    return data


def num_to_latex(raw_value, first_term=True):
    """Convert a complex number to latex code suitable for a ket expression

    Args:
        raw_value: Value to convert
        first_term: If True then generate latex code for the first term in an expression
    Returns:
        String with latex code or None if no term is required
    """
    import sympy  # runtime import

    raw_value = _round_if_close(raw_value)
    value = sympy.nsimplify(raw_value, constants=(sympy.pi,), rational=False)

    if np.abs(value) == 0:
        return None

    element = sympy.latex(value, full_prec=False)
    if isinstance(value, sympy.core.Add):
        # element has two terms
        element = f"({element})"
    else:
        if first_term:
            if element == "1":
                element = ""
            elif element == "-1":
                element = "-"

    if not first_term and not element.startswith('-'):
        element = f"+{element}"

    return element


def _matrix_to_latex(matrix, precision=5, prefix="", max_size=(8, 8)):
    """Latex representation of a complex numpy array (with maximum dimension 2)

    Args:
        matrix (ndarray): The matrix to be converted to latex, must have dimension 2.
        precision (int): For numbers not close to integers, the number of decimal places
                         to round to.
        prefix (str): Latex string to be prepended to the latex, intended for labels.
        max_size (list(```int```)): Indexable containing two integers: Maximum width and maximum
                          height of output Latex matrix (including dots characters). If the
                          width and/or height of matrix exceeds the maximum, the centre values
                          will be replaced with dots. Maximum width or height must be greater
                          than 3.

    Returns:
        str: Latex representation of the matrix

    Raises:
        ValueError: If minimum value in max_size < 3
    """
    if min(max_size) < 3:
        raise ValueError("""Smallest value in max_size must be greater than or equal to 3""")

    out_string = f"\n{prefix}\n"
    out_string += "\\begin{bmatrix}\n"

    def _elements_to_latex(elements):
        # Takes a list of elements (a row) and creates a latex
        # string from it; Each element separated by `&`
        el_string = ""
        for el in elements:
            num_string = num_to_latex(el)  # TODO: make accept precision
            el_string += num_string + " & "
        el_string = el_string[:-2]  # remove trailing ampersands
        return el_string

    def _rows_to_latex(rows, max_width):
        # Takes a list of lists (list of 'rows') and creates a
        # latex string from it
        row_string = ""
        for r in rows:
            if len(r) <= max_width:
                row_string += _elements_to_latex(r)
            else:
                row_string += _elements_to_latex(r[: max_width // 2])
                row_string += "& \\cdots & "
                row_string += _elements_to_latex(r[-max_width // 2 + 1 :])
            row_string += " \\\\\n "
        return row_string

    max_width, max_height = max_size
    if matrix.ndim == 1:
        out_string += _rows_to_latex([matrix], max_width)

    elif len(matrix) > max_height:
        # We need to truncate vertically, so we process the rows at the beginning
        # and end, and add a line of vertical elipse (dots) characters between them
        out_string += _rows_to_latex(matrix[: max_height // 2], max_width)

        if max_width >= matrix.shape[1]:
            out_string += "\\vdots & " * matrix.shape[1]
        else:
            # In this case we need to add the diagonal dots in line with the column
            # of horizontal dots
            pre_vdots = max_width // 2
            post_vdots = max_width // 2 + np.mod(max_width, 2) - 1
            out_string += "\\vdots & " * pre_vdots
            out_string += "\\ddots & "
            out_string += "\\vdots & " * post_vdots

        out_string = out_string[:-2] + "\\\\\n "
        out_string += _rows_to_latex(matrix[-max_height // 2 + 1 :], max_width)

    else:
        out_string += _rows_to_latex(matrix, max_width)
    out_string += "\\end{bmatrix}\n"
    return out_string


def array_to_latex(array, precision=5, prefix="", source=False, max_size=8):
    """Latex representation of a complex numpy array (with dimension 1 or 2)

    Args:
        array (ndarray): The array to be converted to latex, must have dimension 1 or 2 and
                         contain only numerical data.
        precision (int): For numbers not close to integers or common terms, the number of
                         decimal places to round to.
        prefix (str): Latex string to be prepended to the latex, intended for labels.
        source (bool): If ``False``, will return IPython.display.Latex object. If display is
                       ``True``, will instead return the LaTeX source string.
        max_size (list(int) or int): The maximum size of the output Latex array.

            * If list(``int``), then the 0th element of the list specifies the maximum
              width (including dots characters) and the 1st specifies the maximum height
              (also inc. dots characters).
            * If a single ``int`` then this value sets the maximum width _and_ maximum
              height.

    Returns:
        str or IPython.display.Latex: If ``source`` is ``True``, a ``str`` of the LaTeX
            representation of the array, else an ``IPython.display.Latex`` representation of
            the array.

    Raises:
        TypeError: If array can not be interpreted as a numerical numpy array.
        ValueError: If the dimension of array is not 1 or 2.
        MissingOptionalLibraryError: If ``source`` is ``False`` and ``IPython.display.Latex`` cannot be
                     imported.
    """
    try:
        array = np.asarray(array)
        _ = array[0] + 1  # Test first element contains numerical data
    except TypeError as err:
        raise TypeError(
            """array_to_latex can only convert numpy arrays containing numerical data,
        or types that can be converted to such arrays"""
        ) from err

    if array.ndim <= 2:
        if isinstance(max_size, int):
            max_size = (max_size, max_size)
        outstr = _matrix_to_latex(array, precision=precision, prefix=prefix, max_size=max_size)
    else:
        raise ValueError("array_to_latex can only convert numpy ndarrays of dimension 1 or 2")

    if source is False:
        try:
            from IPython.display import Latex
        except ImportError as err:
            raise MissingOptionalLibraryError(
                libname="IPython",
                name="array_to_latex",
                pip_install="pip install ipython",
            ) from err
        return Latex(f"$${outstr}$$")
    else:
        return outstr
