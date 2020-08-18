# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Duration of a circuit instruction.
"""
import warnings


class Duration(float):

    def __new__(self, duration: float, unit: str = None):
        return float.__new__(self, duration)

    def __init__(self, duration: float, unit: str = None):
        float.__init__(duration)
        self.unit = unit

    def in_dt(self, dt: float) -> int:
        res = round(self / dt)
        rounding_error = abs(self - res * dt)
        if rounding_error > 1e-15:
            warnings.warn("Duration is rounded to %d dt = %e [s] from %e [s]"
                          % (res, res * dt, self),
                          UserWarning)
            print("dt=", dt)
        return res
