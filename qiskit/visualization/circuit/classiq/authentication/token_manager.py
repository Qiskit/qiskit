# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Token manager for authenticating to Classiq"""


from typing import Optional

from qiskit.visualization.circuit.classiq.authentication.classiq_auth import Tokens, ClassiqAuth
from qiskit.visualization.circuit.classiq.authentication.device_registration import (
    DeviceRegistrator,
)


class TokenManager:
    """Manager of the tokens for the device used for authentication to Classiq"""

    def __init__(self) -> None:
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None

    def get_access_token(self) -> Optional[str]:
        """Get access token

        Returns:
            str: The access token if exists
        """
        return self._access_token

    def authenticate(self) -> None:
        """Authenticate to the Classiq platform"""
        if self._refresh_token is not None:
            tokens = ClassiqAuth().refresh_access_token(self._refresh_token)
        else:
            tokens = DeviceRegistrator.register(get_refresh_token=True)
        self._save_tokens(tokens)

    def _save_tokens(self, tokens: Tokens) -> None:
        """Save tokens used for authentication

        Args:
            tokens (Tokens): The tokens used for authentication

        """
        self._access_token = tokens.access_token
        if tokens.refresh_token is not None:
            self._refresh_token = tokens.refresh_token
