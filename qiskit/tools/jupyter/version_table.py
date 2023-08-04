# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
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

import time
from sys import modules
from IPython.display import HTML, display
from IPython.core.magic import line_magic, Magics, magics_class
import qiskit
from qiskit.utils import local_hardware_info


@magics_class
class VersionTable(Magics):
    """A class of status magic functions."""

    @line_magic
    def qiskit_version_table(self, line="", cell=None):
        """
        Print an HTML-formatted table with version numbers for Qiskit and its
        dependencies. This should make it possible to reproduce the environment
        and the calculation later on.
        """
        html = "<h3>Version Information</h3>"
        html += "<table>"
        html += "<tr><th>Software</th><th>Version</th></tr>"

        packages = {"qiskit": qiskit.__version__}
        qiskit_modules = {module.split(".")[0] for module in modules.keys() if "qiskit" in module}

        for qiskit_module in qiskit_modules:
            packages[qiskit_module] = getattr(modules[qiskit_module], "__version__", None)

        for name, version in packages.items():
            if version:
                html += f"<tr><td><code>{name}</code></td><td>{version}</td></tr>"

        html += "<tr><th colspan='2'>System information</th></tr>"

        local_hw_info = local_hardware_info()
        sys_info = [
            ("Python version", local_hw_info["python_version"]),
            ("Python compiler", local_hw_info["python_compiler"]),
            ("Python build", local_hw_info["python_build"]),
            ("OS", "%s" % local_hw_info["os"]),
            ("CPUs", "%s" % local_hw_info["cpus"]),
            ("Memory (Gb)", "%s" % local_hw_info["memory"]),
        ]

        for name, version in sys_info:
            html += f"<tr><td>{name}</td><td>{version}</td></tr>"

        html += "<tr><td colspan='2'>%s</td></tr>" % time.strftime("%a %b %d %H:%M:%S %Y %Z")
        html += "</table>"

        return display(HTML(html))
