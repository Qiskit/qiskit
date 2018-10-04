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

-  Create an `IBM Q <https://quantumexperience.ng.bluemix.net>`__ account if
   you haven't already done so
-  Get an API token from the IBM Q website under “My
   Account” > “Personal Access Token”


3.1 Automatically loading credentials
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since Qiskit 0.6, an automatic method that looks for the credentials in several
places can be used for streamlining the setting up of the IBM Q authentication.
This implies that you can set or store your API credentials once after
installation, and when you want to use them, you can simply run:

.. code:: python

    from qiskit import IBMQ

    IBMQ.load_accounts()

This ``IBMQ.load_accounts()`` call performs the automatic loading of the
credentials from several sources, and authenticates against IBM Q, making the
online devices available to your program. Please use one of the following
methods for storing the credentials before calling the automatic registration:

3.1.1 Store API credentials locally
"""""""""""""""""""""""""""""""""""

For most users, storing your API credentials is the most convenient approach.
Your information is stored locally in a configuration file called `qiskitrc`,
and once stored, you can use the credentials without explicitly passing them
to your program.

To store your information, simply run:

.. code:: python

    from qiskit import IBMQ

    IBMQ.save_account('MY_API_TOKEN')


where `MY_API_TOKEN` should be replaced with your token.

If you are on the IBM Q network, you must also pass the `url` 
argument found on your q-console account page to `IBMQ.save_account()`:

.. code:: python

    from qiskit import IBMQ

    IBMQ.save_account('MY_API_TOKEN', url='https://...')


3.1.2 Load API credentials from environment variables
"""""""""""""""""""""""""""""""""""""""""""""""""""""

For more advanced users, it is possible to load API credentials from 
environment variables. Specifically, you can set the following environment
variables:

* `QE_TOKEN`,
* `QE_URL`

Note that if they are present in your environment, they will take precedence
over the credentials stored in disk.

3.1.3 Load API credentials from Qconfig.py
""""""""""""""""""""""""""""""""""""""""""

For compatibility with configurations set for Qiskit versions earlier than 0.6,
the credentials can also be stored in a file called ``Qconfig.py`` placed in
the directory where your program is invoked from. For convenience, we provide
a default version of this file you can use as a reference - using your favorite
editor, create a ``Qconfig.py`` file in the folder of your program with the
following contents:

.. code:: python

    APItoken = 'PUT_YOUR_API_TOKEN_HERE'

    config = {
        'url': 'https://quantumexperience.ng.bluemix.net/api',

        # If you have access to IBM Q features, you also need to fill the "hub",
        # "group", and "project" details. Replace "None" on the lines below
        # with your details from Quantum Experience, quoting the strings, for
        # example: 'hub': 'my_hub'
        # You will also need to update the 'url' above, pointing it to your custom
        # URL for IBM Q.
        'hub': None,
        'group': None,
        'project': None
    }

    if 'APItoken' not in locals():
        raise Exception('Please set up your access token. See Qconfig.py.')

And customize the following lines:

* copy/paste your API token into the space between the quotation marks on the
  first line (``APItoken = 'PUT_YOUR_API_TOKEN_HERE'``).
* if you have access to the IBM Q features, you also need to setup the
  values for your url, hub, group, and project. You can do so by filling the
  ``config`` variable with the values you can find on your IBM Q account
  page.

For example, a valid and fully configured ``Qconfig.py`` file would look like:

.. code:: python

    APItoken = '123456789abc...'

    config = {
        'url': 'https://quantumexperience.ng.bluemix.net/api'
    }

For IBM Q users, a valid and fully configured ``Qconfig.py`` file would look
like:

.. code:: python

    APItoken = '123456789abc...'

    config = {
        'url': 'https://quantumexperience.ng.bluemix.net/api',
        # The following should only be needed for IBM Q users.
        'hub': 'MY_HUB',
        'group': 'MY_GROUP',
        'project': 'MY_PROJECT'
    }

Note that if a ``Qconfig.py`` file is present in your directory, it will take
precedence over the environment variables or the credentials stored in disk.

3.2 Manually loading credentials
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In more complex scenarios or for users that need finer control over multiple
accounts, please note that you can pass the API token and the other parameters
directly to the ``IBMQ.enable_account()`` function, which will ignore the automatic
loading of the credentials and use the arguments directly. For example:

.. code:: python

    from qiskit import IBMQ

    IBMQ.enable_account('MY_API_TOKEN', url='https://my.url')

will try to authenticate using ``MY_API_TOKEN`` and the specified URL,
regardless of the configuration stored in the config file, the environment
variables, or the ``Qconfig.py`` file, if any.

Please refer to the ``qiskit.IBMQ`` documentation for more information about
using multiple credentials.

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



.. _tutorials: https://github.com/Qiskit/qiskit-tutorial
.. _tutorials repository: https://github.com/Qiskit/qiskit-tutorial
.. _documentation for contributors: https://github.com/Qiskit/qiskit-terra/blob/master/.github/CONTRIBUTING.rst
.. _Qconfig.py.default: https://github.com/Qiskit/qiskit-terra/blob/stable/Qconfig.py.default
