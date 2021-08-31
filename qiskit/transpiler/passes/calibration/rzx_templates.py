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

from enum import Enum
from typing import List, Dict

from qiskit.circuit.library.templates.rzx import rzx_zz1, rzx_zz2, rzx_zz3, rzx_yz, rzx_xz, rzx_cy


# pylint: disable=invalid-name
class RZXTemplateMap(Enum):
    """Mapping of instruction name to decomposition template."""
    zz1 = rzx_zz1()
    zz2 = rzx_zz2()
    zz3 = rzx_zz3()
    yz = rzx_yz()
    xz = rzx_xz()
    cy = rzx_cy()


def rzx_templates(template_list: List[str] = None) -> Dict:
    """Convenience function to get the cost_dict and templates for template matching.

    Args:
        template_list: List of instruction names.

    Returns:
        Decomposition templates and cost values.
    """
    if template_list is None:
        template_list = ["zz1", "zz2", "zz3", "yz", "xz", "cy"]

    templates = list(map(lambda gate: RZXTemplateMap[gate].value, template_list))
    cost_dict = {"rzx": 0, "cx": 6, "rz": 0, "sx": 1, "p": 0, "h": 1, "rx": 1, "ry": 1}

    rzx_dict = {"template_list": templates, "user_cost_dict": cost_dict}

    return rzx_dict
