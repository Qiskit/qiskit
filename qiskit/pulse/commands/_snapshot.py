# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Snapshot.
"""

from qiskit.pulse.commands import PulseCommand


class Snapshot(PulseCommand):
    """Snapshot."""

    def __init__(self, label):
        """Create new snapshot command.

        Args:
            label (str): Snapshot label which is used to identify the snapshot in the output.
        """

        super(Snapshot, self).__init__(0)

        self.label = label
