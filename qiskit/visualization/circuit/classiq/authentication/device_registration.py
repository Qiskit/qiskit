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

"""Device registrator for authenticating to Classiq"""


import time
import webbrowser
from datetime import timedelta
from time import sleep
from typing import Dict, Any, Optional

from qiskit.visualization.circuit.classiq.authentication.classiq_auth import Tokens, ClassiqAuth
from qiskit.visualization.circuit.classiq.authentication.auth_exceptions import (
    ClassiqAuthenticationError,
    ClassiqExpiredTokenError,
)


class DeviceRegistrator:
    """Register the device for authentication"""

    _TIMEOUT_ERROR = (
        "Device registration timed out. Please re-initiate the flow and "
        "authorize the device within the timeout."
    )
    _TIMEOUT_SEC: float = timedelta(minutes=15).total_seconds()

    @classmethod
    def register(cls, get_refresh_token: bool = True) -> Tokens:
        """Register the device for authentication

        Args:
            get_refresh_token (bool): Whether to refresh the token for authentication

        Returns:
            Tokens: the tokens used to authenticate the device
        """

        classiq_auth_client = ClassiqAuth()
        data = classiq_auth_client.get_device_data(get_refresh_token=get_refresh_token)

        print(f"Your user code is: {data['user_code']}")
        verification_url = data["verification_uri_complete"]
        print(f"If a browser doesn't automatically open, please visit the url: {verification_url}")
        webbrowser.open(verification_url)
        timeout = min(data["expires_in"], cls._TIMEOUT_SEC)
        return cls._poll_tokens(
            auth0_client=classiq_auth_client,
            device_code=data["device_code"],
            interval=data["interval"],
            timeout=timeout,
            get_refresh_token=get_refresh_token,
        )

    @classmethod
    def _handle_ready_data(cls, data: Dict[str, Any], get_refresh_token: bool) -> Tokens:
        """Convert data from the http response to Tokens object

        Args:
            data (Dict[str, Any]): the data with the tokens received in the http response
            get_refresh_token (bool): True if refresh token was requested

        Returns:
            Tokens: A Token instance with the tokens received for the device

        """

        access_token: Optional[str] = data.get("access_token")
        # If refresh token was not requested, this would be None
        refresh_token: Optional[str] = data.get("refresh_token")

        if access_token is None or (get_refresh_token is True and refresh_token is None):
            raise ClassiqAuthenticationError("Token generation failed for unknown reason.")

        return Tokens(access_token=access_token, refresh_token=refresh_token)

    @classmethod
    def _poll_tokens(
        cls,
        auth0_client: ClassiqAuth,
        device_code: str,
        interval: int,
        timeout: float,
        get_refresh_token: bool = True,
    ) -> Tokens:
        """poll tokens for device registration

        Args:
            auth0_client (ClassiqAuth): Client used for the authentication to Classiq
            device_code (str): The device code
            interval (int): Time to wait between tokens polling failed attempts
            timeout (float): Max time to try polling the tokens until success
            get_refresh_token (bool): Whether to refresh the token for authentication

        Returns:
            Tokens: the tokens used to authenticate the device

        """

        start = time.monotonic()

        while time.monotonic() <= start + timeout:
            sleep(interval)
            data = auth0_client.poll_tokens(device_code=device_code)
            error_code: Optional[str] = data.get("error")
            if error_code is None:
                return cls._handle_ready_data(data, get_refresh_token)
            elif error_code in ("slow_down", "authorization_pending"):
                pass
            elif error_code == "expired_token":
                raise ClassiqExpiredTokenError(cls._TIMEOUT_ERROR)
            elif error_code == "access_denied":
                error_description: str = data.get("error_description")
                raise ClassiqAuthenticationError(error_description)
            else:
                raise ClassiqAuthenticationError(
                    f"Device registration failed with an unknown error: {error_code}."
                )
        raise ClassiqAuthenticationError(cls._TIMEOUT_ERROR)
