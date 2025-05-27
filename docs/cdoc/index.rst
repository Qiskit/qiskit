===========================
Qiskit C API (``qiskit.h``)
===========================

The Qiskit C API is a low level interface to the core data model of Qiskit. It
is designed to provide a high performance interface to Qiskit for compiled
languages and provides a defined ABI to the internal Rust data model that is
used to create the Python API. There are two expected modes of operation for
the C API:

* A standalone shared library for creating and working with Qiskit objects from
  compiled languages without a runtime dependency on Python, and
* For building Python extensions that are using Qiskit but interface directly
  with the Rust objects from the extension code without using Python for better
  performance.

To get started, see `Install the Qiskit C API <https://docs.quantum.ibm.com/guides/install-c-api>`_.
To combine the C API with custom Python extensions, see
`Extend Qiskit in Python with C <https://docs.quantum.ibm.com/guides/c-extension-for-python>`_.

As this interface is still new in Qiskit it should be considered experimental
and the interface might change between minor version releases.

---------------
Quantum Circuit
---------------

.. toctree::
   :maxdepth: 1

   qk-circuit

-------------------
Quantum information
-------------------

.. toctree::
   :maxdepth: 1

   qk-obs
   qk-obs-term
   qk-bit-term
   qk-exit-code
