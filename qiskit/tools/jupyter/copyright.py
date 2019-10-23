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
# pylint: disable=unused-argument

"""A module for monitoring backends."""

import datetime
from IPython.display import HTML, display               # pylint: disable=import-error
from IPython.core.magic import (line_magic,             # pylint: disable=import-error
                                Magics, magics_class)


@magics_class
class Copyright(Magics):
    """A class of status magic functions.
    """
    @line_magic
    def qiskit_copyright(self, line='', cell=None):
        """A Jupyter magic function return qiskit copyright
        """
        now = datetime.datetime.now()

        html = "<div style='width: 100%; background-color:#d5d9e0;"
        html += "padding-left: 10px; padding-bottom: 10px; padding-right: 10px; padding-top: 5px'>"
        html += "<h3>This code is a part of Qiskit</h3>"
        html += "<p>&copy; Copyright IBM 2017, %s.</p>" % now.year
        html += "<p>This code is licensed under the Apache License, Version 2.0. You may<br>"
        html += "obtain a copy of this license in the LICENSE.txt file in the root directory<br> "
        html += "of this source tree or at http://www.apache.org/licenses/LICENSE-2.0."

        html += "<p>Any modifications or derivative works of this code must retain this<br>"
        html += "copyright notice, and modified files need to carry a notice indicating<br>"
        html += "that they have been altered from the originals.</p>"
        html += "</div>"
        return display(HTML(html))
