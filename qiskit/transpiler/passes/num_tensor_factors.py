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

"""Calculate the number of tensor factors of a DAG circuit."""

from qiskit.transpiler.basepasses import AnalysisPass


class NumTensorFactors(AnalysisPass):
    """Calculate the number of tensor factors of a DAG circuit.

    The result is saved in ``property_set['num_tensor_factors']`` as an integer.
    """

    def run(self, dag):
        """Run the NumTensorFactors pass on `dag`."""
        self.property_set['num_tensor_factors'] = dag.num_tensor_factors()
