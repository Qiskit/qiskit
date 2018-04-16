from abc import ABC, abstractmethod

class BaseJob(ABC):
    """Class to handle asynchronous jobs"""
    
    @abstractmethod
    def __init__(self):
        """Initializes and initates the asynchronous job"""
        pass

    @abstractmethod
    def result(self):
        """
        Returns:
            qiskit.Result:
        """
        pass

    @abstractmethod
    def cancelled(self):
        """
        Returns:
            bool: True if call was successfully cancelled
        """
        pass

    @abstractmethod
    def done(self):
        """
        Returns:
            bool: True if call was successfully cancelled or finished.
        """
        pass    

    @abstractmethod
    def status(self):
        """
        Returns:
            str or code: user defined
        """
        pass

    @abstractmethod    
    def abort(self):
        """
        Attempt to cancel job.
        Returns:
            bool: True if job can be cancelled, else False.
        """
        pass

    @abstractmethod
    def running(self):
        """
        Returns:
            bool: True if job is currently running.
        """
        pass
