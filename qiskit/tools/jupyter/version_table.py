# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
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

import sys
import time
from IPython.display import HTML, display               # pylint: disable=import-error
from IPython.core.magic import (line_magic,             # pylint: disable=import-error
                                Magics, magics_class)
import qiskit
from qiskit.util import local_hardware_info


@magics_class
class VersionTable(Magics):
    """A class of status magic functions.
    """
    @line_magic
    def qiskit_version_table(self, line='', cell=None):
        """
        Print an HTML-formatted table with version numbers for Qiskit and its
        dependencies. This should make it possible to reproduce the environment
        and the calculation later on.
        """
        html = "<h3>Version Information</h3>"
        html += "<table>"
        html += "<tr><th>Qiskit Software</th><th>Version</th></tr>"

        packages = []
        qver = qiskit.__qiskit_version__

        packages.append(('Qiskit', qver['qiskit']))
        packages.append(('Terra', qver['qiskit-terra']))
        packages.append(('Aer', qver['qiskit-aer']))
        packages.append(('Ignis', qver['qiskit-ignis']))
        packages.append(('Aqua', qver['qiskit-aqua']))
        packages.append(('IBM Q Provider', qver['qiskit-ibmq-provider']))

        for name, version in packages:
            html += "<tr><td>%s</td><td>%s</td></tr>" % (name, version)

        html += "<tr><th>System information</th></tr>"

        local_hw_info = local_hardware_info()
        sys_info = [("Python", sys.version),
                    ("OS", "%s" % local_hw_info['os']),
                    ("CPUs", "%s" % local_hw_info['cpus']),
                    ("Memory (Gb)", "%s" % local_hw_info['memory'])
                    ]

        for name, version in sys_info:
            html += "<tr><td>%s</td><td>%s</td></tr>" % (name, version)

        html += "<tr><td colspan='2'>%s</td></tr>" % time.strftime(
            '%a %b %d %H:%M:%S %Y %Z')
        html += "</table>"

        return display(HTML(html))
