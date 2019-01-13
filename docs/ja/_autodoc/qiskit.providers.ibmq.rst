qiskit.providers.ibmq package
=============================


.. automodule:: qiskit.providers.ibmq
    :members: least_busy
    :undoc-members:
    :show-inheritance:
    
    Submodules
    ----------

    .. toctree::
       :maxdepth: 1
   


       qiskit.providers.ibmq.ibmqaccounterror
       qiskit.providers.ibmq.ibmqbackend
       qiskit.providers.ibmq.ibmqjob
       qiskit.providers.ibmq.ibmqprovider
       qiskit.providers.ibmq.ibmqsingleprovider

    Subpackages
    -----------

    .. toctree::
       :maxdepth: 1

       qiskit.providers.ibmq.api
       qiskit.providers.ibmq.credentials

    Exceptions
    ----------


    .. list-table::
    
       * - :exc:`QiskitError <qiskit.qiskiterror.QiskitError>`
         - Base class for errors raised by the Qiskit.
    

    Classes
    -------


    .. list-table::
    
       * - :class:`IBMQBackend <qiskit.providers.ibmq.ibmqbackend.IBMQBackend>`
         - Backend class interfacing with the Quantum Experience remotely.
       * - :class:`IBMQJob <qiskit.providers.ibmq.ibmqjob.IBMQJob>`
         - Represent the jobs that will be executed on IBM-Q simulators and real devices.
       * - :class:`IBMQProvider <qiskit.providers.ibmq.ibmqprovider.IBMQProvider>`
         - Provider for remote IBMQ backends with admin features.
    




    Functions
    ---------


    .. list-table::
    
       * - :func:`least_busy <qiskit.providers.ibmq.least_busy>`
         - Return the least busy available backend for those that have a `pending_jobs` in their `status`.
    