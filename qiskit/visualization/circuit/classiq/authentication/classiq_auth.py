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

"""Classiq authentication helper"""


from dataclasses import dataclass
from typing import Any, Dict, Optional

from qiskit.utils import optionals as _optionals
from .auth_exceptions import ClassiqAuthenticationError


@dataclass
class Tokens:
    """Contain the token data required to authenticate to Classiq"""

    access_token: str
    refresh_token: Optional[str]


@_optionals.HAS_REQUESTS.require_in_instance
class ClassiqAuth:
    """Client Used to authenticate to the Classiq platform"""

    _BASE_URL = "https://auth.classiq.io"
    _AUDIENCE = "https://cadmium-be"
    _CLIENT_ID = "f6721qMOVoDAOVkzrv8YaWassRKSFX6Y"
    _CONTENT_TYPE = "application/x-www-form-urlencoded"
    _HEADERS = {"content-type": _CONTENT_TYPE}

    def __init__(self):
        import requests as _requests

        self._requests = _requests

    def _make_request(
        self,
        url: str,
        payload: Dict[str, str],
    ) -> Dict[str, Any]:
        """Utility method to make http requests

        Args:
            url (str): the url to send the http request to
            payload (Dict[str, str]): the payload of the desired request

        Returns:
            Dict[str, Any]: the response of the http request

        """

        response = self._requests.post(url=self._BASE_URL + url, data=payload, timeout=60)
        data = response.json()
        if not response.ok and not data.get("error") in ("slow_down", "authorization_pending"):
            raise ClassiqAuthenticationError(
                f"Authentication failed with code: {response.status_code}: {data.get('error')}"
            )
        return data

    def get_device_data(self, get_refresh_token: bool = True) -> Dict[str, Any]:
        """get the device data for the device registration authentication flow

        Args:
             get_refresh_token (bool): Whether to refresh the token for authentication

        Returns:
            Dict[str, Any]: The http response with the device data

        """
        payload = {"client_id": self._CLIENT_ID, "audience": self._AUDIENCE}
        if get_refresh_token:
            payload["scope"] = "offline_access"

        return self._make_request(
            url="/oauth/device/code",
            payload=payload,
        )

    def poll_tokens(self, device_code: str) -> Dict[str, Any]:
        """poll the tokens required for registering the device

        Args:
            device_code (str): the code of the device from which we want to get tokens to.

        Returns:
            Dict[str, Any]: the http response with the token data.
        """
        payload = {
            "client_id": self._CLIENT_ID,
            "device_code": device_code,
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        }

        return self._make_request(
            url="/oauth/token",
            payload=payload,
        )

    def refresh_access_token(self, refresh_token: str) -> Tokens:
        """Refreshes access token

        Args:
            refresh_token (str): the new access token

        Returns:
            Tokens: Tokens instance with the new token
        """
        payload = {
            "client_id": self._CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
        data = self._make_request(
            url="/oauth/token",
            payload=payload,
        )

        return Tokens(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token", None),
        )
