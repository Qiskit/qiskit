# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Convenience function to load RZXGate based templates.
"""

from typing import List

from qiskit.circuit.library.templates.rzx import rzx_zz1, rzx_zz2, rzx_zz3, rzx_yz, rzx_xz, rzx_cy


def rzx_templates(template_list: List[str] = None):
    """
    Convenience function to get the cost_dict and
    templates for template matching.
    """

    if template_list is None:
        template_list = ["zz1", "zz2", "zz3", "yz", "xz", "cy"]

    templates = []
    if "zz1" in template_list:
        templates.append(rzx_zz1())
    if "zz2" in template_list:
        templates.append(rzx_zz2())
    if "zz3" in template_list:
        templates.append(rzx_zz3())
    if "yz" in template_list:
        templates.append(rzx_yz())
    if "xz" in template_list:
        templates.append(rzx_xz())
    if "cy" in template_list:
        templates.append(rzx_cy())

    cost_dict = {"rzx": 0, "cx": 6, "rz": 0, "sx": 1, "p": 0, "h": 1, "rx": 1, "ry": 1}

    rzx_dict = {"template_list": templates, "user_cost_dict": cost_dict}

    return rzx_dict
