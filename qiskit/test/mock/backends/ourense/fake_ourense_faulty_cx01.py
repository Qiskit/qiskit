# -*- coding: utf-8 -*-

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
Fake Ourense device (5 qubit). With a faulty CX(Q0, Q1)
"""

import os
import json

from qiskit.providers.models import BackendProperties
from .fake_ourense import FakeOurense


class FakeOurenseFaultyCX01(FakeOurense):
    """A fake 5 qubit backend, with a faulty CX(Q1, Q3)
         0 (↔) 1 ↔ 3 ↔ 4
               ↕
               2
    """

    def properties(self):
        """Returns a snapshot of device properties as recorded on 8/30/19.
        Sets the gate CX(Q0, Q1) (and symmetric) as non-operational.
        """
        dirname = os.path.dirname(__file__)
        filename = "props_ourense.json"
        with open(os.path.join(dirname, filename), "r") as f_prop:
            props = json.load(f_prop)
        for qubit in props['gates']:
            if qubit['name'] == 'cx0_1' or qubit['name'] == 'cx1_0':
                qubit['parameters'].append({"date": "2000-01-01 00:00:00Z",
                                            "name": "operational",
                                            "unit": "",
                                            "value": 0})
        return BackendProperties.from_dict(props)
