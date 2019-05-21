
######################
Contributing to Qiskit
######################

Qiskit is an open-source project committed to bringing quantum computing to people of all
backgrounds. This page describes how you can join the Qiskit community in this goal.


****************
Where Things Are
****************

The code for Qiskit is located in the `Qiskit GitHub organization <https://github.com/Qiskit>`_,
where you can find the individual projects that make up Qiskit, including

* `Qiskit Terra <https://github.com/Qiskit/qiskit-terra>`__
* `Qiskit Aer <https://github.com/Qiskit/qiskit-aer>`__
* `Qiskit Ignis <https://github.com/Qiskit/qiskit-ignis>`__
* `Qiskit Aqua <https://github.com/Qiskit/qiskit-aqua>`__
* `Qiskit Chemistry <https://github.com/Qiskit/qiskit-chemistry>`__
* `Qiskit IBMQ Provider <https://github.com/Qiskit/qiskit-ibmq-provider>`__
* `Qiskit Tutorials <https://github.com/Qiskit/qiskit-tutorials>`__
* `Qiskit Documentation <https://github.com/Qiskit/qiskit/tree/master/docs>`__


****************
Getting Started
****************

Learn how members of the Qiskit community

* `Relate to one another <https://github.com/Qiskit/qiskit/blob/master/.github/CODE_OF_CONDUCT.md>`_
* `Discuss ideas <https://qiskit.slack.com/>`_
* `Get help when we're stuck <https://quantumcomputing.stackexchange.com/questions/tagged/qiskit>`_
* `Stay informed of news in the community <https://medium.com/qiskit>`_
* `Keep a consistent style <https://www.python.org/dev/peps/pep-0008>`_
* `Work together on GitHub <https://github.com/Qiskit/qiskit/blob/master/.github/CONTRIBUTING.md>`_
* :ref:`Build Qiskit packages from source <install_install_from_source_label>`


**********************************
Writing and Building Documentation
**********************************

Qiskit documentation is shaped by the `docs as code
<https://www.writethedocs.org/guide/docs-as-code/>`_ philosophy and follows the
`IBM style guidelines
<https://www.ibm.com/developerworks/library/styleguidelines/>`_.

The `published documentation <https://qiskit.org/documentation/index.html>`_ is
built from the master branch of `Qiskit/qiskit/docs
<https://github.com/Qiskit/qiskit/tree/master/docs>`_ using `Sphinx
<http://www.sphinx-doc.org/en/master/>`_.

You can build a local copy of the documentation from your local clone of the
`Qiskit/qiskit` repository as follows:

1. Clone `Qiskit/qiskit` (or your personal fork).

2. `Install Sphinx <http://www.sphinx-doc.org/en/master/usage/installation.html>`_.

3. Install the `Material Design HTML Theme for Sphinx` by running the following
   in a terminal window:

   .. code-block:: sh

     pip install sphinx_materialdesign_theme

4. Build the documentation by navigating to your local clone of `Qiskit/qiskit`
   and running the following command in a terminal window:

   .. code-block:: sh

     make doc

As you make changes to your local RST files, you can update your
HTML files by navigating to `/doc/` and running the following in a terminal
window:

.. code-block:: sh

  make html

This will build a styled, HTML version of your local documentation repository
in the subdirectory `/docs/_build/html/`.

.. _install_install_from_source_label:

*******************
Install from Source
*******************

Installing the elements from source allows you to access the most recently
updated version of Qiskit instead of using the version in the Python Package
Index (PyPI) repository. This will give you the ability to inspect and extend
the latest version of the Qiskit code more efficiently.

When installing the elements and components from source, by default their
``development`` version (which corresponds to the ``master`` git branch) will
be used, as opposed to the ``stable`` version (which contains the same codebase
as the published ``pip`` packages). Since the ``development`` versions of an
element or component usually includes new features and changes, in general they
require using the ``development`` version of the rest of the items as well.

.. note::

  The Terra and Aer packages both require a compiler to build from source before
  you can install. Ignis, Aqua, Qiskit Chemistry, and the IBM Q provider backend
  do not require a compiler.

Installing elements from source requires the following order of installation to
prevent getting versions of elements that may be lower than those desired if the
pip version is behind the source versions:

#. qiskit-terra
#. qiskit-ibmq-provider (if wanting to connect to the IBM Q devices or online
   simulator)
#. qiskit-aer
#. qiskit-ignis
#. qiskit-aqua

To work with several components and elements simultaneously, use the following
steps for each element.

The following steps show the installation process for Ignis.

1. Clone the Qiskit element repository.

.. code-block:: sh

    git clone https://github.com/Qiskit/qiskit-ignis.git

2. Create a virtual development environment.

.. code-block:: sh

    conda create -y -n QiskitDevenv python=3
    conda activate QiskitDevenv

3. Install the package in `editable mode <https://pip.pypa.io/en/stable/
   reference/pip_install/#editable-installs>`_ from the root directory of the
   repository. The following example shows the installation for Ignis.

.. code:: sh

  pip install -e qiskit-ignis

Install Terra from Source
--------------------------------
Installing from source requires that you have a c++ compiler on your system that supports
c++-11.  On most Linux platforms, the necessary GCC compiler is already installed.

Install a compiler for MacOS
""""""""""""""""""""""""""""

If you use Apple OSX, you can install the Clang compiler by installing XCode.
Check if you have XCode and clang installed by opening a terminal window and entering the
following.

.. code:: sh

  clang --version

Install XCode and clang by using the following command.

.. code:: sh

    xcode-select --install

Install a compiler for Windows
""""""""""""""""""""""""""""""
On Windows, it is easiest to install the Visual C++ compiler from the
`Build Tools for Visual Studio 2017 <https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017>`_.
You can instead install Visual Studio version 2015 or 2017, making sure to select the
options for installing the C++ compiler.

Install Qiskit Terra
^^^^^^^^^^^^^^^^^^^^^
1. Clone the Terra repository.

.. code:: sh

  git clone https://github.com/Qiskit/qiskit-terra.git

2. Cloning the repository creates a local folder called ``qiskit-terra``.

.. code:: sh

  cd qiskit-terra

3. Install the Python requirements libraries from your ``qiskit-terra`` directory.

.. code:: sh

    pip install cython

* If you want to run tests or linting checks, install the developer requirements.

.. code:: sh

    pip install -r requirements-dev.txt

4. Install the Qiskit modules.

* If you want to only install ``qiskit-terra`` onto your system.

.. code:: sh

    pip install .


* To get the examples working, install and run them with the following commands.

.. code:: sh

    pip install -e .
    python examples/python/using_qiskit_terra_level_0.py


After you've installed Terra, you can install Aer as an add-on to run additional simulators.

* `Qiskit Aer <https://github.com/Qiskit/qiskit-aer/blob/master/.github/
  CONTRIBUTING.md>`__
* `Qiskit Ignis <https://github.com/Qiskit/qiskit-ignis/blob/master/.github/
  CONTRIBUTING.md>`_
* `Qiskit Aqua <https://github.com/Qiskit/qiskit-aqua/blob/master/.github/
  CONTRIBUTING.rst>`__
* `Qiskit Chemistry <https://github.com/Qiskit/qiskit-chemistry/blob/master/
  .github/CONTRIBUTING.rst>`__
* `Qiskit IBMQ Provider <https://github.com/Qiskit/qiskit-ibmq-provider/blob/
  master/.github/CONTRIBUTING.rst>`__
