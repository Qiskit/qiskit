"""This module implements the abstract base class for backend modules.

To create add-on backend modules subclass the Backend class in this module. 
Doing so requires that the required backend interface is implemented.
"""
from abc import ABC, abstractmethod

class BaseBackend(ABC):
    
    @abstractmethod
    def __init__(self, qobj):
        """initialize

        This method should initialize the module and raise a FileNotFoundError 
        exception if a component of the moduel is not available. 

        Args:
            qobj (dict): qobj dictionary
        
        Raises:
            FileNotFoundError if backend executable is not available.
        """
        self._qobj = qobj
        self._configuration = None # IMPLEMENT for your backend

    @abstractmethod
    def run(self):
        pass

    @property
    @abstractmethod
    def configuration(self):
        """Return backend configuration"""
        pass

    @configuration.setter
    @abstractmethod
    def configuration(self, configuration):
        """Set backend configuration"""
        pass
