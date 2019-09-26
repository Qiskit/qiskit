.. _faq:

==========================
Frequently Asked Questions
==========================

**Q: How should I cite Qiskit in my research?**

**A:** Please cite Qiskit by using the included `BibTeX file
<https://raw.githubusercontent.com/Qiskit/qiskit/master/Qiskit.bib>`__.

|

**Q: Why do I receive the error message** ``Error: Instance of QuantumCircuit has no
member`` **when adding gates to a circuit?**

**A:** This is a pylint error, which is a Linter for Python. Linters analyze
code for potential errors, and they throw errors when they find
potentially erroneous code. However, this error should not prevent your
code from compiling or running, so there is no need to worry. The error
message can be disabled by adding the following line above the code that
is causing the error:

.. code:: python

  #pylint: disable=no-member

|

**Q: Why do my results from real devices differ from my results from the simulator?**

**A:** The simulator runs jobs as though is was in an ideal environment; one
without noise or decoherence. However, when jobs are run on the real devices
there is noise from the environment and decoherence, which causes the qubits
to behave differently than what is intended.

|

**Q: Why do I receive the error message,** ``No Module 'qiskit'`` **when using Jupyter Notebook?**

**A:** If you used ``pip install qiskit`` and set up your virtual environment in
Anaconda, then you may experience this error when you run a tutorial
in Jupyter Notebook. If you have not installed Qiskit or set up your
virtual environment, you can follow the
`installation steps <https://qiskit.org/documentation/install.html#install>`__.

The error is caused when trying to import the Qiskit package in an
environment where Qiskit is not installed. If you launched Jupyter Notebook
from the Anaconda-Navigator, it is possible that Jupyter Notebook is running
in the base (root) environment, instead of in your virtual
environment. Choose a virtual environment in the Anaconda-Navigator from the
**Applications on** dropdown menu. In this menu, you can see
all of the virtual environments within Anaconda, and you can
select the environment where you have Qiskit installed to launch Jupyter
Notebook.

|

**Q: Why am I getting a compilation error while installing ``qiskit``?**

**A:** Qiskit depends on a number of other open source Python packages, which
are automatically installed when doing ``pip install qiskit``. Depending on
your system's platform and Python version, is it possible that a particular
package does not provide a pre-built binary wheel for your system, and in those
cases ``pip`` will attempt to compile the package from source, which in turn
might require some extra dependencies that need to be installed manually.

If the output of ``pip install qiskit`` contains similar lines to:

.. code:: sh

  Failed building wheel for SOME_PACKAGE
  ...
  build/temp.linux-x86_64-3.5/_openssl.c:498:30: fatal error
  compilation terminated.
  error: command 'x86_64-linux-gnu-gcc' failed with exit status 1

please check the documentation of the package that failed to install (in the
example code, ``SOME_PACKAGE``) for information on how to install the libraries
needed for compiling from source. For example:

* ``cryptography``: https://cryptography.io/en/latest/installation/#building-cryptography-on-linux
