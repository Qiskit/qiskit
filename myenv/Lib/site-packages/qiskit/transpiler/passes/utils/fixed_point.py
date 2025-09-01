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

"""Check if a property reached a fixed point."""

from copy import deepcopy

from qiskit.transpiler.basepasses import AnalysisPass


class FixedPoint(AnalysisPass):
    """Check if a property reached a fixed point.

    A dummy analysis pass that checks if a property reached a fixed point.
    The result is saved in ``property_set['<property>_fixed_point']``
    as a boolean.
    """

    def __init__(self, property_to_check):
        """FixedPoint initializer.

        Args:
            property_to_check (str): The property to check if a fixed point was reached.
        """
        super().__init__()
        self._property = property_to_check

    def run(self, dag):
        """Run the FixedPoint pass on `dag`."""
        current_value = self.property_set[self._property]
        fixed_point_previous_property = f"_fixed_point_previous_{self._property}"

        if self.property_set[fixed_point_previous_property] is None:
            self.property_set[f"{self._property}_fixed_point"] = False
        else:
            fixed_point_reached = self.property_set[fixed_point_previous_property] == current_value
            self.property_set[f"{self._property}_fixed_point"] = fixed_point_reached

        self.property_set[fixed_point_previous_property] = deepcopy(current_value)
