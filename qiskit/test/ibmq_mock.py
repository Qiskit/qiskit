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

"""Mock functions for qiskit.IBMQ."""

from unittest.mock import MagicMock
import qiskit
from qiskit.providers import fake_provider as backend_mocks


def mock_get_backend(backend):
    """Replace qiskit.IBMQ with a mock that returns a single backend.

    Note this will set the value of qiskit.IBMQ to a MagicMock object. It is
    intended to be run as part of docstrings with jupyter-example in a hidden
    cell so that later examples which rely on ibmq devices so that the docs can
    be built without requiring configured credentials. If used outside of this
    context be aware that you will have to manually restore qiskit.IBMQ the
    value to qiskit.providers.ibmq.IBMQ after you finish using your mock.

    Args:
        backend (str): The class name as a string for the fake device to
            return from the mock IBMQ object. For example, FakeVigo.
    Raises:
        NameError: If the specified value of backend
    """
    mock_ibmq = MagicMock()
    mock_provider = MagicMock()
    if not hasattr(backend_mocks, backend):
        raise NameError(
            "The specified backend name is not a valid mock from qiskit.providers.fake_provider."
        )
    fake_backend = getattr(backend_mocks, backend)()
    mock_provider.get_backend.return_value = fake_backend
    mock_ibmq.get_provider.return_value = mock_provider
    qiskit.IBMQ = mock_ibmq
