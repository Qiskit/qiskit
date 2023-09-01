# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name,missing-docstring
# pylint: disable=attribute-defined-outside-init


from qiskit.transpiler.passes import TimeUnitConversion


class SchedulingPassBenchmarks:

    params = ([5, 10, 20], [500, 1000])
    param_names = ["n_qubits", "depth"]
    timeout = 300

    def time_time_unit_conversion_pass(self, _, __):
        TimeUnitConversion(self.durations).run(self.dag)
