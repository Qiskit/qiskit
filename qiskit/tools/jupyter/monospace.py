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
# pylint: disable=unused-argument

"""A Jupyter magic to choose a real monospaced fonts, if available."""

from IPython.display import HTML, display
from IPython.core.magic import line_magic, Magics, magics_class


@magics_class
class MonospacedOutput(Magics):
    """A class for setting "Courier New" for output code."""

    @line_magic
    def monospaced_output(self, line="", cell=None):
        """A Jupyter magic function to set "Courier New" for output code."""
        html = """<style type='text/css'>
        code, kbd, pre, samp {font-family: Courier New,monospace;line-height: 1.1;}</style>"""
        display(HTML(html))
