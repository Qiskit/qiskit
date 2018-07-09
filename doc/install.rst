======================
Installation and setup
======================

Installation
============

1. Dependencies
---------------

To use Qiskit you'll need to have installed at least
`Python 3.5 or later <https://www.python.org/downloads/>`__.
`Jupyter Notebooks <https://jupyter.readthedocs.io/en/latest/install.html>`__
is also recommended for interacting with
`tutorials`_.

For this reason we recommend installing `Anaconda 3 <https://www.continuum.io/downloads>`__
python distribution, which already comes with all these dependencies pre-installed.


2. Installation
---------------

The recommended way to install Qiskit is by using the PIP tool (Python
package manager):

.. code:: sh

    pip install qiskit

This will install the latest stable release along with all the dependencies.


.. _qconfig-setup:

3. Configure your API token and QE credentials
----------------------------------------------

-  Create an `IBM Q
   experience <https://quantumexperience.ng.bluemix.net>`__ account if
   you haven't already done so
-  Get an API token from the IBM Q experience website under “My
   Account” > “Personal Access Token”

3.1 Store API credentials locally
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For most users, storing your API credentials is most easily done locally.
Your information is stored locally in a configuration file called `qiskitrc`.
To store your information, simply run:

.. code:: python

    from qiskit import store_credentials

    store_credentials(`MY_API_TOKEN`)

where `MY_API_TOKEN` should be replaced with your token.

If you are on the IBM Q network, you must also pass `url`,
`hub`, `group`, and `project` arguments to `store_credentials`.

To register your credentials with QISKit, simply run:

.. code:: python

    from qiskit import register

    register()


3.2 Load API credentials from environment variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For more advanced users, it is possible to load API credentials from 
environmental variables.  Specifically, one may set `QE_TOKEN`,
`QE_URL`, `QE_HUB`, `QE_GROUP`, and `QE_PROJECT`.  These can then be registered 
with QISKit:

.. code:: python

    from qiskit import register

    register()

3.3 Load API credentials from Qconfig.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The API token can be loaded from a file called 
``Qconfig.py``. For convenience, we provide a default version of 
this file that you can use as a reference: `Qconfig.py.default`_. 
After downloading that file, copy it into the folder where you 
will be invoking the SDK (on Windows, replace ``cp`` with ``copy``):

.. code:: sh

    cp Qconfig.py.default Qconfig.py

Open your ``Qconfig.py``, remove the ``#`` from the beginning of the API
token line, and copy/paste your API token into the space between the
quotation marks on that line. Save and close the file.

For example, a valid and fully configured ``Qconfig.py`` file would look like:

.. code:: python

    APItoken = '123456789abc...'

    config = {
        'url': 'https://quantumexperience.ng.bluemix.net/api'
    }

If you have access to the IBM Q features, you also need to setup the
values for your hub, group, and project. You can do so by filling the
``config`` variable with the values you can find on your IBM Q account
page.

For example, a valid and fully configured ``Qconfig.py`` file for IBM Q
users would look like:

.. code:: python

    APItoken = '123456789abc...'

    config = {
        'url': 'https://quantumexperience.ng.bluemix.net/api',
        # The following should only be needed for IBM Q users.
        'hub': 'MY_HUB',
        'group': 'MY_GROUP',
        'project': 'MY_PROJECT'
    }

If the `Qconfig.py` is in the current working directory, then it can be
automatrically registered with QISKit:

.. code:: python

    from qiskit import register

    register()

Install Jupyter-based tutorials
===============================

The Qiskit project provides you a collection of tutorials in the form of Jupyter
notebooks, which are essentially web pages that contain "cells" of embedded
Python code. Please refer to the `tutorials repository`_ for detailed
instructions.


Troubleshooting
===============

The installation steps described on this document assume familiarity with the
Python environment on your setup (for example, standard Python, ``virtualenv``
or Anaconda). Please consult the relevant documentation for instructions
tailored to your environment.

Depending on the system and setup, appending "sudo -H" before the
``pip install`` command could be needed:

.. code:: sh

    pip install -U --no-cache-dir qiskit


For additional troubleshooting tips, see the `Qiskit troubleshooting page
<https://github.com/Qiskit/qiskit-terra/wiki/QISKit-Troubleshooting>`_
on the project's GitHub wiki.

.. _tutorials: https://github.com/Qiskit/qiskit-tutorial
.. _tutorials repository: https://github.com/Qiskit/qiskit-tutorial
.. _documentation for contributors: https://github.com/Qiskit/qiskit-terra/blob/master/.github/CONTRIBUTING.rst
.. _Qconfig.py.default: https://github.com/Qiskit/qiskit-terra/blob/stable/Qconfig.py.default
