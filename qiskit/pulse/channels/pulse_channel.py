from abc import ABCMeta, abstractmethod

from qiskit.pulse import commands


class PulseChannel(metaclass=ABCMeta):
    """Pulse channel."""

    supported = None
    prefix = None

    @abstractmethod
    def __init__(self, index: int):
        self._index = index

    @property
    def index(self) -> int:
        return self._index

    @property
    def name(self) -> str:
        return '%s%d' % (self.__class__.prefix, self._index)

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self._index)

    def __eq__(self, other):
        """Two channels are the same if they are of the same type, and have the same index.

        Args:
            other (PulseChannel): other PulseChannel

        Returns:
            bool: are self and other equal.
        """
        if type(self) is type(other) and \
                self._index == other._index:
            return True
        return False


class AcquireChannel(PulseChannel):
    """Acquire channel."""

    supported = commands.Acquire
    prefix = 'a'

    def __init__(self, index):
        """Create new acquire channel.

        Args:
            index (int): Index of the channel.
        """
        super().__init__(index)


class MemorySlot(PulseChannel):
    """Memory slot."""

    supported = commands.Acquire
    prefix = 'm'

    def __init__(self, index):
        """Create new memory slot.

        Args:
            index (int): Index of the channel.
        """
        super().__init__(index)


class RegisterSlot(PulseChannel):
    """Classical resister slot channel."""

    supported = commands.Acquire
    prefix = 'r'

    def __init__(self, index):
        """Create new register slot.

        Args:
            index (int): Index of the channel.
        """
        super().__init__(index)
