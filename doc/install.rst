======================
Installation and setup
======================

Installation
============

1. Get the tools
----------------

To use QISKit you'll need to have installed at least
`Python 3.5 or later <https://www.python.org/downloads/>`__ and
`Jupyter Notebooks <https://jupyter.readthedocs.io/en/latest/install.html>`__
(recommended for interacting with the tutorials).

For this reason we recommend installing `Anaconda 3 <https://www.continuum.io/downloads>`__
python distribution, which already comes with all these dependencies pre-installed.

if you are a Mac OS X user, you will find Xcode useful: https://developer.apple.com/xcode/

if you are willing to contribute to QISKit or just wanted to extend it, you
should install Git too: https://git-scm.com/download/.


2. PIP Install
--------------

The fastest way to install QISKit is by using the PIP tool (Python package manager):

.. code:: sh

    pip install qiskit

3. Repository Install
---------------------

Other common option is to clone the QISKit SDK repository on your local machine,
and change into the cloned directory:

-  If you have Git installed, run the following commands:

.. code:: sh

    git clone https://github.com/QISKit/qiskit-sdk-py
    cd qiskit-sdk-py

- If you don't have Git installed, click the "Clone or download"
  button on the `QISKit SDK GitHub repo <https://github.com/QISKit/qiskit-sdk-py>`__, unzip the file if
  needed and finally change into the unziped directory.

3.1 Setup the environment
^^^^^^^^^^^^^^^^^^^^^^^^^

To use QISKit as standalone library, install all the dependencies:

.. code:: sh

    # Depending on the system and setup, appending "sudo -H" before this command could be needed.
    pip install -r requirements.txt

To get the tutorials working set up an Anaconda environment for working
with QISKit, and install the required dependencies:

-  If running either Linux or Mac OS X with Xcode, simply run the
   following command:

.. code:: sh

    make env

-  If running on Mac OS X without Xcode, run the following set of commands:

.. code:: sh

    conda create -y -n QISKitenv python=3 pip scipy
    activate QISKitenv
    pip install -r requirements.txt
    
-  If running on Windows, make sure to execute an Anaconda Prompt and run
   the following command:

.. code:: sh

    .\make env


4. Configure your API token
---------------------------

-  Create an `IBM Q
   experience <https://quantumexperience.ng.bluemix.net>`__ account if
   you haven't already done so
-  Get an API token from the IBM Q experience website under “My
   Account” > “Personal Access Token”
-  You will insert your API token in a file called Qconfig.py. First
   copy the default version of this file from the tutorial folder to the
   main SDK folder (on Windows, replace ``cp`` with ``copy``):

.. code:: sh

    cp Qconfig.py.default Qconfig.py

-  Open your Qconfig.py, remove the ``#`` from the beginning of the API
   token line, and copy/paste your API token into the space between the
   quotation marks on that line. Save and close the file.

Install Jupyter-based tutorials
===============================

The QISKit project provides you a collection of tutorials in the form of Jupyter 
notebooks, which are essentially web pages that contain "cells" of embedded 
Python code. To run a cell, click on it and hit ``Shift+Enter`` or use the 
toolbar at the top of the page. Any output from a cell is displayed 
immediately below it on the page. In most cases, the cells on each page must
be run in sequential order from top to bottom in order to avoid errors. To get
started with the tutorials, follow the instructions below.

1.1 Install standalone
----------------------
- download the tutorials: https://github.com/QISKit/qiskit-tutorial/archive/master.zip
- uncompress the zip file
- in the terminal/command-line and into the folder "qiskit-tutorial-master" execute:

.. code:: sh

    jupyter notebook index.ipynb

Please refer to the
`qiskit-tutorial repository <https://github.com/QISKit/qiskit-tutorial>`__
for further instructions on how to execute them.
    

FAQ
===

If you upgrade the dependencies and get the error below, try the fix
shown below the error:

- Depending on the system and setup, appending "sudo -H" before this command could be needed.

.. code:: sh

    pip install -U --no-cache-dir IBMQuantumExperience
    
- Fix: run the command below

.. code:: sh

    curl https://bootstrap.pypa.io/ez_setup.py -o - | python

For additional troubleshooting tips, see the QISKit troubleshooting page
on the project's GitHub wiki
