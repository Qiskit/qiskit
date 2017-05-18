"""
Exception for errors raised by the QISKit SDK.

Author: Andrew Cross
"""


class QISKitException(Exception):
    """Base class for errors raised by the QISKit SDK."""

    def __init__(self, *message):
        """Set the error message."""
        self.message = ' '.join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)
