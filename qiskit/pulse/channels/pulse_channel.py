from abc import ABCMeta, abstractmethod


class PulseChannel(metaclass=ABCMeta):
    """Pulse Channel."""

    @abstractmethod
    def __init__(self, index: int):
        self.index = index

    def __str__(self):
        return '%s%d' % (self.__class__.prefix, self.index)

    @property
    def name(self):
        return str(self)

    def __eq__(self, other):
        """Two channels are the same if they are of the same type, and have the same index.

        Args:
            other (PulseChannel): other PulseChannel

        Returns:
            bool: are self and other equal.
        """
        if type(self) is type(other) and \
                self.index == other.index:
            return True
        return False
