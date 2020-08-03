# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from fractions import Fraction
import numpy as np
import math


def _num_to_latex(num, precision=5):
    """Takes a complex number as input and returns a latex representation

        Args:
            num (numerical): The number to be converted to latex.
            precision (int): If the real or imaginary parts of num are not close
                             to an integer, the number of decimal places to round to

        Returns:
            str: Latex representation of num
    """
    # Result is combination of maximum 4 strings in the form:
    #     {common_facstring} ( {realstring} {operation} {imagstring}i )
    # common_facstring: A common factor between the real and imaginary part
    # realstring: The real part (inc. a negative sign if applicable)
    # operation: The operation between the real and imaginary parts ('+' or '-')
    # imagstring: Absolute value of the imaginary parts (i.e. not inc. any negative sign).
    # This function computes each of these strings and combines appropriately.

    r = np.real(num)
    i = np.imag(num)
    common_factor = None

    # try to factor out common terms in imaginary numbers
    if np.isclose(abs(r), abs(i)) and not np.isclose(r, 0):
        common_factor = abs(r)
        r = r/common_factor
        i = i/common_factor

    common_terms = {
        1/math.sqrt(2): '\\tfrac{1}{\\sqrt{2}}',
        1/math.sqrt(3): '\\tfrac{1}{\\sqrt{3}}',
        math.sqrt(2/3): '\\sqrt{\\tfrac{2}{3}}',
        math.sqrt(3/4): '\\sqrt{\\tfrac{3}{4}}',
        1/math.sqrt(8): '\\tfrac{1}{\\sqrt{8}}'
    }

    def _proc_value(val):
        # This function converts a real value to a latex string
        # First, see if val is close to an integer:
        val_mod = np.mod(val, 1)
        if (np.isclose(val_mod, 0) or np.isclose(val_mod, 1)):
            # If so, return that integer
            return str(int(np.round(val)))
        # Otherwise, see if it matches one of the common terms
        for term, latex_str in common_terms.items():
            if np.isclose(abs(val), term):
                if val > 0:
                    return latex_str
                else:
                    return "-" + latex_str
        # try to factorise val nicely
        frac = Fraction(val).limit_denominator()
        num, denom = frac.numerator, frac.denominator
        if num + denom < 20:
            # If fraction is 'nice' return
            if val > 0:
                return "\\tfrac{%i}{%i}" % (abs(num), abs(denom))
            else:
                return "-\\tfrac{%i}{%i}" % (abs(num), abs(denom))
        else:
            # Failing everything else, return val as a decimal
            return "{:.{}f}".format(val, precision).rstrip("0")

    # Get string (or None) for common factor between real and imag
    if common_factor is not None:
        common_facstring = _proc_value(common_factor)
    else:
        common_facstring = None

    # Get string for real part
    realstring = _proc_value(r)

    # Get string for both imaginary part and operation between real and imaginary parts
    if i > 0:
        operation = "+"
        imagstring = _proc_value(i)
    else:
        operation = "-"
        imagstring = _proc_value(-i)
    if imagstring == "1":
        imagstring = ""  # Don't want to return '1i', just 'i'

    # Now combine the strings appropriately:
    if imagstring == "0":
        return realstring  # realstring already contains the negative sign (if needed)
    if realstring == "0":
        # imagstring needs the negative sign adding
        if operation == "-":
            return "-{}i".format(imagstring)
        else:
            return "{}i".format(imagstring)
    if common_facstring is not None:
        return "{}({} {} {}i)".format(common_facstring, realstring, operation, imagstring)
    else:
        return "{} {} {}i".format(realstring, operation, imagstring)


def _vector_to_latex(vector, precision=5, pretext="", max_size=16):
    """Latex representation of a complex numpy array (with dimension 1)

        Args:
            vector (ndarray): The vector to be converted to Latex, must have dimension 1.
            precision (int): For numbers not close to integers, the number of decimal places
            to round to.
            pretext (str): Latex string to be prepended to the latex, intended for labels.
            max_size (int): The maximum number of elements present in the output Latex
                            (including the vertical dots character). If the vector is larger
                            than this, the centre elements will be replaced with vertical dots.
                            Must be greater than 3.

        Returns:
            str: Latex representation of the vector, wrapped in $$

        Raises:
            ValueError: If max_size < 3
    """
    if max_size < 3:
        raise ValueError("""max_size must be greater than or equal to 3""")

    out_string = "\n$$\n{}\n".format(pretext)
    out_string += "\\begin{bmatrix}\n"

    def _elements_to_latex(elements):
        # Returns the latex representation of each numerical element, separated by "\\\\\n"
        el_string = ""
        for e in elements:
            num_string = _num_to_latex(e, precision=precision)
            el_string += num_string + " \\\\\n "
        return el_string

    if len(vector) <= max_size:
        out_string += _elements_to_latex(vector)
    else:
        out_string += _elements_to_latex(vector[:max_size//2])
        out_string += "\\vdots \\\\\n "
        out_string += _elements_to_latex(vector[-max_size//2+1:])
    if len(vector) != 0:
        out_string = out_string[:-4] + "\n"  # remove trailing characters
    out_string += "\\end{bmatrix}\n$$"
    return out_string


def _matrix_to_latex(matrix, precision=5, pretext="", max_size=(8, 8)):
    """Latex representation of a complex numpy array (with dimension 2)

        Args:
            matrix (ndarray): The matrix to be converted to latex, must have dimension 2.
            precision (int): For numbers not close to integers, the number of decimal places
                             to round to.
            pretext (str): Latex string to be prepended to the latex, intended for labels.
            max_size (list(```int```)): Indexable containing two integers: Maximum width and maximum
                              height of output Latex matrix (including dots characters). If the
                              width and/or height of matrix exceeds the maximum, the centre values
                              will be replaced with dots. Maximum width or height must be greater
                              than 3.

        Returns:
            str: Latex representation of the matrix, wrapped in $$

        Raises:
            ValueError: If minimum value in max_size < 3
    """
    if min(max_size) < 3:
        raise ValueError("""Smallest value in max_size must be greater than or equal to 3""")

    out_string = "\n$$\n{}\n".format(pretext)
    out_string += "\\begin{bmatrix}\n"

    def _elements_to_latex(elements):
        el_string = ""
        for e in elements:
            num_string = _num_to_latex(e, precision=precision)
            el_string += num_string + " & "
        el_string = el_string[:-2]  # remove trailing ampersands
        return el_string

    def _rows_to_latex(rows, max_width):
        row_string = ""
        for r in rows:
            if len(r) <= max_width:
                row_string += _elements_to_latex(r)
            else:
                row_string += _elements_to_latex(r[:max_width//2])
                row_string += "& \\cdots & "
                row_string += _elements_to_latex(r[-max_width//2+1:])
            row_string += " \\\\\n "
        return row_string

    max_width, max_height = max_size
    if len(matrix) > max_height:
        out_string += _rows_to_latex(matrix[:max_height//2], max_width)

        # number of vertical dots preceding and trailing the diagonal dots:
        pre_vdots = max_width//2
        post_vdots = max_width//2 + np.mod(max_width, 2) - 1

        out_string += "\\vdots & "*pre_vdots + "\\ddots & " + "\\vdots & "*post_vdots
        out_string = out_string[:-2] + "\\\\\n "
        out_string += _rows_to_latex(matrix[-max_height//2+1:], max_width)

    else:
        out_string += _rows_to_latex(matrix, max_width)
    out_string += "\\end{bmatrix}\n$$"
    return out_string
