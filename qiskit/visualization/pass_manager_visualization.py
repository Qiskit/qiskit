# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Visualization function for a pass manager. Passes are grouped based on their
flow controller, and coloured based on the type of pass.
"""

# Temporary import from 0.22.0 to be deprecated in future
# pylint: disable=unused-import
from .plots.pass_manager_visualization import pass_manager_drawer
