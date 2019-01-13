qiskit.providers package
========================


.. automodule:: qiskit.providers
    
    
    Submodules
    ----------

    .. toctree::
       :maxdepth: 1
   


       qiskit.providers.basebackend
       qiskit.providers.basejob
       qiskit.providers.baseprovider
       qiskit.providers.exceptions
       qiskit.providers.jobstatus
       qiskit.providers.providerutils

    Subpackages
    -----------

    .. toctree::
       :maxdepth: 1

       qiskit.providers.builtinsimulators
       qiskit.providers.ibmq
       qiskit.providers.legacysimulators
       qiskit.providers.models

    Exceptions
    ----------


    .. list-table::
    
       * - :exc:`JobError <qiskit.providers.exceptions.JobError>`
         - Base class for errors raised by jobs.
       * - :exc:`JobTimeoutError <qiskit.providers.exceptions.JobTimeoutError>`
         - Base class for timeout errors raised by jobs.
       * - :exc:`QiskitBackendNotFoundError <qiskit.providers.exceptions.QiskitBackendNotFoundError>`
         - Base class for errors raised while looking up for a backend.
    

    Classes
    -------


    .. list-table::
    
       * - :class:`BaseBackend <qiskit.providers.basebackend.BaseBackend>`
         - Base class for backends.
       * - :class:`BaseJob <qiskit.providers.basejob.BaseJob>`
         - Class to handle asynchronous jobs
       * - :class:`BaseProvider <qiskit.providers.baseprovider.BaseProvider>`
         - Base class for a backend provider.
       * - :class:`JobStatus <qiskit.providers.jobstatus.JobStatus>`
         - Class for job status enumerated type.
    