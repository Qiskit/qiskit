"""
Exception for errors raised by unroller backends.

Author: Andrew Cross
"""


class BackendException(Exception):
    """Base class for errors raised by unroller backends."""

    def __init__(self, *message):
        """Set the error message."""
        self.message = ' '.join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)
