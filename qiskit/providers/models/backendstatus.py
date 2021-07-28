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

"""Class for backend status."""

import html
from qiskit.exceptions import QiskitError


class BackendStatus:
    """Class representing Backend Status."""

    def __init__(
        self,
        backend_name: str,
        backend_version: str,
        operational: bool,
        pending_jobs: int,
        status_msg: str,
    ):
        """Initialize a BackendStatus object

        Args:
            backend_name: The backend's name
            backend_version: The backend's version of the form X.Y.Z
            operational: True if the backend is operational
            pending_jobs: The number of pending jobs on the backend
            status_msg: The status msg for the backend

        Raises:
            QiskitError: If the backend version is in an invalid format
        """
        self.backend_name = backend_name
        self.backend_version = backend_version
        self.operational = operational
        if pending_jobs < 0:
            raise QiskitError("Pending jobs must be >=0")
        self.pending_jobs = pending_jobs
        self.status_msg = status_msg

    @classmethod
    def from_dict(cls, data):
        """Create a new BackendStatus object from a dictionary.

        Args:
            data (dict): A dictionary representing the BaseBakend to create.
                         It will be in the same format as output by
                         :func:`to_dict`.

        Returns:
            BackendStatus: The BackendStatus from the input dictionary.
        """
        return cls(**data)

    def to_dict(self):
        """Return a dictionary format representation of the BackendStatus.

        Returns:
            dict: The dictionary form of the QobjHeader.
        """
        return self.__dict__

    def __eq__(self, other):
        if isinstance(other, BackendStatus):
            if self.__dict__ == other.__dict__:
                return True
        return False

    def _repr_html_(self) -> str:
        """Return html representation of the object

        Returns:
            Representation used in Jupyter notebook and other IDE's that call the method

        """
        rpr = self.__repr__()
        html_code = (
            f"<pre>{html.escape(rpr)}</pre>"
            f"<b>name</b>: {self.backend_name}<br/>"
            f"<b>version</b>: {self.backend_version},"
            f" <b>pending jobs</b>: {self.pending_jobs}<br/>"
            f"<b>status</b>: {self.status_msg}<br/>"
        )

        return html_code
