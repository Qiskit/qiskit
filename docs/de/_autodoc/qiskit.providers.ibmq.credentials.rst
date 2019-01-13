qiskit.providers.ibmq.credentials package
=========================================


.. automodule:: qiskit.providers.ibmq.credentials
    :members: discover_credentials
    :undoc-members:
    :show-inheritance:
    
    Submodules
    ----------

    .. toctree::
       :maxdepth: 1
   


       
       
       
       qiskit.providers.ibmq.credentials.credentials

    Exceptions
    ----------


    .. list-table::
    
       * - :exc:`QiskitError <qiskit.qiskiterror.QiskitError>`
         - Base class for errors raised by the Qiskit.
    

    Classes
    -------


    .. list-table::
    
       * - :class:`Credentials <qiskit.providers.ibmq.credentials.credentials.Credentials>`
         - IBM Q account credentials.
       * - :class:`OrderedDict <collections.OrderedDict>`
         - Dictionary that remembers insertion order
    




    Functions
    ---------


    .. list-table::
    
       * - :func:`discover_credentials <qiskit.providers.ibmq.credentials.discover_credentials>`
         - Automatically discover credentials for IBM Q.
       * - :func:`read_credentials_from_environ <qiskit.providers.ibmq.credentials._environ.read_credentials_from_environ>`
         - Read the environment variables and return its credentials.
       * - :func:`read_credentials_from_qconfig <qiskit.providers.ibmq.credentials._qconfig.read_credentials_from_qconfig>`
         - Read a `QConfig.py` file and return its credentials.
       * - :func:`read_credentials_from_qiskitrc <qiskit.providers.ibmq.credentials._configrc.read_credentials_from_qiskitrc>`
         - Read a configuration file and return a dict with its sections.
       * - :func:`store_credentials <qiskit.providers.ibmq.credentials._configrc.store_credentials>`
         - Store the credentials for a single account in the configuration file.
    