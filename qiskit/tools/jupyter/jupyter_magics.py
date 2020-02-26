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

"""A module of magic functions"""

from IPython.core import magic_arguments                           # pylint: disable=import-error
from IPython.core.magic import (line_magic, Magics, magics_class)  # pylint: disable=import-error
import qiskit
from qiskit.tools.events.progressbar import TextProgressBar
from .progressbar import HTMLProgressBar


@magics_class
class ProgressBarMagic(Magics):
    """A class of progress bar magic functions.
    """
    @line_magic
    @magic_arguments.magic_arguments()
    @magic_arguments.argument(
        '-t',
        '--type',
        type=str,
        default='html',
        help="Type of progress bar, 'html' or 'text'."
    )
    def qiskit_progress_bar(self, line='', cell=None):  # pylint: disable=unused-argument
        """A Jupyter magic function to generate progressbar.
        """
        args = magic_arguments.parse_argstring(self.qiskit_progress_bar, line)
        if args.type == 'html':
            pbar = HTMLProgressBar()
        elif args.type == 'text':
            pbar = TextProgressBar()
        else:
            raise qiskit.QiskitError('Invalid progress bar type.')

        return pbar
