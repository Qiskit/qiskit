"""
Exception for errors raised by the QISKit SDK.

Author: Andrew Cross
"""


class QISKitException(Exception):
    """Base class for errors raised by the QISKit SDK."""

    def __init__(self, *msg):
        """Set the error message."""
        self.msg = ' '.join(msg)

    def __str__(self):
        """Return the message."""
        return repr(self.msg)
