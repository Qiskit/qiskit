======================
Installation and setup
======================

Installation
============

1. Dependencies
---------------

To use QISKit you'll need to have installed at least
`Python 3.5 or later <https://www.python.org/downloads/>`__.
`Jupyter Notebooks <https://jupyter.readthedocs.io/en/latest/install.html>`__
is also recommended for interacting with
`tutorials`_.

For this reason we recommend installing `Anaconda 3 <https://www.continuum.io/downloads>`__
python distribution, which already comes with all these dependencies pre-installed.


2. Installation
---------------

The recommended way to install QISKit is by using the PIP tool (Python
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
-  The API token needs to be placed in a file called ``Qconfig.py``. For
   convenience, we provide a default version of this file that you
   can use as a reference: `Qconfig.py.default`_. After downloading that
   file, copy it into the folder where you will be invoking the SDK (on
   Windows, replace ``cp`` with ``copy``):

.. code:: sh

    cp Qconfig.py.default Qconfig.py

-  Open your ``Qconfig.py``, remove the ``#`` from the beginning of the API
   token line, and copy/paste your API token into the space between the
   quotation marks on that line. Save and close the file.

For example, a valid and fully configured ``Qconfig.py`` file would look like:

.. code:: python

    APItoken = '123456789abc...'

    config = {
        'url': 'https://quantumexperience.ng.bluemix.net/api'
    }

-  If you have access to the IBM Q features, you also need to setup the
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


Install Jupyter-based tutorials
===============================

The QISKit project provides you a collection of tutorials in the form of Jupyter 
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


For additional troubleshooting tips, see the `QISKit troubleshooting page
<https://github.com/QISKit/qiskit-sdk-py/wiki/QISKit-Troubleshooting>`_
on the project's GitHub wiki.

.. _tutorials: https://github.com/QISKit/qiskit-tutorial
.. _tutorials repository: https://github.com/QISKit/qiskit-tutorial
.. _documentation for contributors: https://github.com/QISKit/qiskit-sdk-py/blob/master/CONTRIBUTING.rst
.. _Qconfig.py.default: https://github.com/QISKit/qiskit-sdk-py/blob/stable/Qconfig.py.default