Installation and setup
======================

1. Get the tools
----------------

To use QISKit Python version you'll need to have installed [Python 3 or later]
(https://www.python.org/downloads/) and [Jupyter Notebooks]
(https://jupyter.readthedocs.io/en/latest/install.html) 
(recommended to interact with tutorials). 

For this reason e recomend to use [Anaconda 3](https://www.continuum.io/downloads) 
python distribution for install all of this dependencies.

if you are a Mac OS X users will find Xcode useful: https://developer.apple.com/xcode/

if you nnedd to get the QISKit code to extend it you can download Git: https://git-scm.com/download/.


2. PIP Install 
--------------

the fast way to install QISKit is using PIP tool (Python package manager):

.. code:: sh

    pip install qiskit

3 Repository Install
---------------------

Other common option is clone the QISKit SDK repository and navigate to its 
folder on your local machine:

-  If you have Git installed, run the following commands:

.. code:: sh

    git clone https://github.com/QISKit/qiskit-sdk-py
    cd qiskit-sdk-py

-  If you don't have Git installed, click the "Clone or download" button
   at the URL shown in the git clone command, unzip the file if needed,
   then navigate to that folder in a terminal window.

3.1 Setup the environment
-------------------------

To use as a library install the dependencies:

.. code:: sh

    # Depending on the system and setup to append "sudo -H" before could be needed.
    pip install -r requires.txt

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
    pip install -r requires.txt
    
-  If running on Windows, make sure to execute an Anaconda Prompt and run
   the following command:

.. code:: sh

    .\make env


4. Configure your API token
---------------------------

-  Create an `IBM Quantum
   Experience <https://quantumexperience.ng.bluemix.net>`__ account if
   you haven't already done so
-  Get an API token from the Quantum Experience website under “My
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

The QISKit project provide you collection of tutorials in the form of Jupyter 
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
- in the terminal/commandline and into the folder "qiskit-tutorial-master" execute:

.. code:: sh

    jupyter notebook index.ipynb

1.2 Install into the QISKit folder
----------------------------------

-  If running either Linux or Mac OS X with Xcode, simply run the
   following command from the QISKit SDK folder:

.. code:: sh

    make install-tutorials

    make run-tutorials
    
-  If running on Windows, make sure you are running an Anaconda Prompt,
   and then run the following commands from the QISKit SDK folder:

    - download the tutorials: https://github.com/QISKit/qiskit-tutorial/archive/master.zip
    - uncompress the zip file
    - move the content into the a new "tutorials" folder in the QISKit folder

.. code:: sh

    .\make run-tutorials
    

FAQ
===

If you upgrade the dependencies and get the error below, try the fix
shown below the error:

- Depending on the system and setup to append "sudo -H" before could be needed.

.. code:: sh

    pip install --upgrade IBMQuantumExperience
    
- Fix: run the command below

.. code:: sh

    curl https://bootstrap.pypa.io/ez_setup.py -o - | python

For additional troubleshooting tips, see the QISKit troubleshooting page
on the project's GitHub wik
