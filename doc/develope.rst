Develope and extend QISKit
==========================

Installation and setup
----------------------

1. Get the tools
~~~~~~~~~~~~~~~~

You'll need:

-  Install `Python 3 <https://docs.python.org/3/using/index.html>`__.
-  `Jupyter <http://jupyter.readthedocs.io/en/latest/install.html>`__
   client is needed to run the tutorials, not to use as a library.
-  Mac OS X users will find Xcode useful:
   https://developer.apple.com/xcode/
-  For Windows users we highly recommend to install `Anaconda 3 <https://www.continuum.io/downloads#windows>`_
-  Optionally download Git: https://git-scm.com/download/.

2. Get the code
~~~~~~~~~~~~~~~

Clone the QISKit SDK repository and navigate to its folder on your local
machine:

-  If you have Git installed, run the following commands:

.. code:: sh

    git clone https://github.com/QISKit/qiskit-sdk-py
    cd qiskit-sdk-py

-  If you don't have Git installed, click the "Clone or download" button
   at the URL shown in the git clone command, unzip the file if needed,
   then navigate to that folder in a terminal window.

3. Setup the environment
~~~~~~~~~~~~~~~~~~~~~~~~

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


Anaconda Installation 
~~~~~~~~~~~~~~~~~~~~~

For those who would prefer to use Anaconda, you can use the following QISKit install process instead:

### Dependencies > NEEDS REVIEW

* [Anaconda](https://www.continuum.io/downloads) (**QUESTION: What version is needed?**)
* [Jupyter Notebooks](https://jupyter.readthedocs.io/en/latest/install.html) (recommended to interact with tutorials)

A basic understanding of quantum information is also very helpful when interacting with QISKit. If you're new to quantum, Start with our [User Guides](https://github.com/QISKit/ibmqx-user-guides)!

### User Installation > NEEDS CONTENT

**ANACONDA INSTALL INSTRUCTIONS TO GO HERE**

## Getting Started

#### NEEDS REVIEW