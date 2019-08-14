
######################
Contributing to Qiskit
######################

Qiskit is an open-source project committed to bringing quantum computing to people of all
backgrounds. This page describes how you can join the Qiskit community in this goal.


****************
Where Things Are
****************

The code for Qiskit is located in the `Qiskit GitHub organization <https://github.com/Qiskit>`__,
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

* `Relate to one another <https://github.com/Qiskit/qiskit/blob/master/CODE_OF_CONDUCT.md>`__
* `Discuss ideas <https://qiskit.slack.com/>`__
* `Get help when we're stuck <https://quantumcomputing.stackexchange.com/questions/tagged/qiskit>`__
* `Stay informed of news in the community <https://medium.com/qiskit>`__
* `Keep a consistent style <https://www.python.org/dev/peps/pep-0008>`__
* `Work together on GitHub <https://github.com/Qiskit/qiskit/blob/master/CONTRIBUTING.md>`__
* :ref:`Build Qiskit packages from source <install_install_from_source_label>`


**********************************
Writing and Building Documentation
**********************************

Qiskit documentation is shaped by the `docs as code
<https://www.writethedocs.org/guide/docs-as-code/>`__ philosophy and follows the
`IBM style guidelines
<https://www.ibm.com/developerworks/library/styleguidelines/>`__.

You can find templates for writing new documentation topics, such as helping users understand a
concept, perform a task, or find technical specifications in the
`Docs Templates folder <https://github.com/Qiskit/qiskit/tree/master/.github/DOCS_TEMPLATES>`__.

The `published documentation <https://qiskit.org/documentation/index.html>`__ is
built from the master branch of `Qiskit/qiskit/docs
<https://github.com/Qiskit/qiskit/tree/master/docs>`__ using `Sphinx
<http://www.sphinx-doc.org/en/master/>`__.

You can build a local copy of the documentation from your local clone of the
`Qiskit/qiskit` repository as follows:

1. Clone `Qiskit/qiskit` (or your personal fork).

2. `Install Sphinx <http://www.sphinx-doc.org/en/master/usage/installation.html>`__.

3. Install the required dependencies by running the following
   in a terminal window:

   .. code-block:: sh

      pip install -r requirements-dev.txt

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

**********************
Installing from Source
**********************

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
#. qiskit-chemistry

To work with several components and elements simultaneously, use the following
steps for each element.

.. note::

   Due to the use of namespace packaging in Python, care must be taken in how you
   install packages. If you're planning to install any element from source do not
   use the ``qiskit`` meta-package. Also follow this guide and use a separate virtual
   environment for development. If you do choose to mix an existing installation
   with your development refer to:
   https://github.com/pypa/sample-namespace-packages/blob/master/table.md
   for the set of of combinations for installation methods that work together.

The following steps show the installation process for Ignis.

1. Clone the Qiskit element repository.

   .. code-block:: sh

       git clone https://github.com/Qiskit/qiskit-ignis.git

2. Create a virtual development environment.

   .. code-block:: sh

       conda create -y -n QiskitDevenv python=3
       conda activate QiskitDevenv

3. Install the package in `editable mode <https://pip.pypa.io/en/stable/
   reference/pip_install/#editable-installs>`__ from the root directory of the
   repository. The following example shows the installation for Ignis.

   .. code:: sh

      pip install -e qiskit-ignis

Installing Terra from Source
----------------------------
Installing from source requires that you have a c++ compiler on your system that supports
c++-11.

.. tabs::

   .. tab:: Compiler for Linux

      On most Linux platforms, the necessary GCC compiler is already installed.

   .. tab:: Compiler for macOS

      If you use macOS, you can install the Clang compiler by installing XCode.
      Check if you have XCode and clang installed by opening a terminal window and entering the
      following.

      .. code:: sh

            clang --version

      Install XCode and clang by using the following command.

      .. code:: sh

            xcode-select --install

   .. tab:: Compiler for Windows

      On Windows, it is easiest to install the Visual C++ compiler from the
      `Build Tools for Visual Studio 2017 <https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017>`__.
      You can instead install Visual Studio version 2015 or 2017, making sure to select the
      options for installing the C++ compiler.

Once the compilers are installed, you are ready to install Qiskit Terra.

1. Clone the Terra repository.

   .. code:: sh

      git clone https://github.com/Qiskit/qiskit-terra.git

2. Cloning the repository creates a local folder called ``qiskit-terra``.

   .. code:: sh

      cd qiskit-terra

3. Install the Python requirements libraries from your ``qiskit-terra`` directory.

   .. code:: sh

      pip install cython

.. tabs::

   .. tab:: Run Tests

      If you want to run tests or linting checks, install the developer requirements.

      .. code:: sh

         pip install -r requirements-dev.txt

      4. Install ``qiskit-terra``.

         .. code:: sh

            pip install .

   .. tab:: Editable Mode

      If you want to install it in editable mode, meaning that code changes to the
      project don't require a reinstall to be applied you can do this with:

      .. code:: sh

          pip install -e .

   .. tab:: Run Examples

      You can then run the code examples working after installing terra. You can
      run the example with the following command.

      .. code:: sh

         python examples/python/using_qiskit_terra_level_0.py


After you've installed Terra, you can install Aer as an add-on to run additional simulators.

.. note::

    If you do not intend to install any other components qiskit-terra will
    emit a ``RuntimeWarning`` warning that both qiskit-aer and
    qiskit-ibmq-provider are not installed. This is done because the more
    common case is to have users who intend to use the additional elements
    but not realize their not installed, or have the installation fail. If
    you wish to suppress these warnings this is easy to do by adding::

        import warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                                module='qiskit')

    before any ``qiskit`` imports in your code. That will suppress just the
    warning about the missing qiskit-aer and qiskit-ibmq-provider, but still
    display any other warnings from qiskit or other packages.

Installing IBMQ Provider from Source
------------------------------------

1. Clone the qiskit-ibmq-provider repository.

   .. code:: sh

      git clone https://github.com/Qiskit/qiskit-ibmq-provider.git

2. Cloning the repository creates a local directory called ``qiskit-ibmq-provider``.

   .. code:: sh

      cd qiskit-ibmq-provider

3. If you want to run tests or linting checks, install the developer requirements.
   This is not required to install or use the qiskit-ibmq-provider package when
   installing from source.

   .. code:: sh

      pip install -r requirements-dev.txt

4. Install qiskit-ibmq-provider

   .. code:: sh

      pip install .

If you want to install it in editable mode, meaning that code changes to the
project don't require a reinstall to be applied you can do this with:

.. code:: sh

    pip install -e .

Installing Aer from Source
--------------------------

1. Clone the Aer repository.

   .. code:: sh

      git clone https://github.com/Qiskit/qiskit-aer

2. Install build requirements.

   .. code:: sh

      pip install cmake scikit-build cython

After this the steps to install Aer depend on which operating system you are
using. Since Aer is a compiled C++ program with a python interface there are
non-python dependencies for building the Aer binary which can't be installed
universally depending on operating system.

.. tabs::

   .. tab:: Linux

      3. Install compiler requirements.

         Building Aer requires a C++ compiler and development headers

         If you're using Ubuntu>=16.04 or an equivalent Debian Linux distribution
         you can install this with:

         .. code:: sh

            sudo apt install build-essential

      4. Install OpenBLAS development headers.

         If you're using Ubuntu>=16.04 or an equivalent Debian Linux distribution,
         you can install this with:

         .. code:: sh

            sudo apt install libopenblas-dev


      5. Build and install qiskit-aer directly

         If you have pip <19.0.0 installed and your environment doesn't require a
         custom build options you can just run:

         .. code:: sh

            cd qiskit-aer
            pip install .

         This will both build the binaries and install Aer.

         Alternatively if you have a newer pip installed, or have some custom requirement
         you can build a python wheel manually.

         .. code:: sh

            cd qiskit-aer
            python ./setup.py bdist_wheel

         If you need to set a custom option during the wheel build you can refer to
         :ref:`aer_wheel_build_options`.

         After you build the python wheel it will be stored in the ``dist/`` dir in the
         Aer repository. The exact version will depend

         .. code:: sh

             cd dist
             pip install qiskit_aer-*.whl

         The exact filename of the output wheel file depends on the current version of
         Aer under development.

   .. tab:: macOS

      3. Install dependencies.

         To use the `Clang <https://clang.llvm.org/>`__ compiler on macOS, you need to install
         an extra library for supporting `OpenMP <https://www.openmp.org/>`__.  You can use `brew <https://brew.sh/>`__
         to install this and other dependencies.

         .. code:: sh

            brew install libomp

         You then also have to install a BLAS implementation, `OpenBLAS <https://www.openblas.net/>`__
         is the default choice.

         .. code:: sh

            brew install openblas

         You also need to have ``Xcode Command Line Tools`` installed.

         .. code:: sh

            xcode-select --install

      4. Build and install qiskit

         If you have pip <19.0.0 installed and your environment doesn't require a
         custom build options you can just run:

         .. code:: sh

            cd qiskit-aer
            pip install .

         This will both build the binaries and install Aer.

         Alternatively if you have a newer pip installed, or need to set custom options
         for your environment you can build a python wheel manually.

         .. code:: sh

            cd qiskit-aer
            python ./setup.py bdist_wheel

         If you need to set a custom option during the wheel build you can refer to
         :ref:`aer_wheel_build_options`.

         After you build the python wheel it will be stored in the ``dist/`` dir in the
         Aer repository. The exact version will depend

         .. code:: sh

            cd dist
            pip install qiskit_aer-*.whl

         The exact filename of the output wheel file depends on the current version of
         Aer under development.

   .. tab:: Windows

      On Windows you need to use `Anaconda3 <https://www.anaconda.com/distribution/#windows>`__
      or `Miniconda3 <https://docs.conda.io/en/latest/miniconda.html>`__ to install all the
      dependencies.

      3. Install compiler requirements

         .. code:: sh

            conda install --update-deps vs2017_win-64 vs2017_win-32 msvc_runtime

      4. Install binary and build dependencies

         .. code:: sh

            conda install --update-deps -c conda-forge -y openblas cmake

      5. Install the package

         If you have pip <19.0.0 installed you can just run

         .. code:: sh

            cd qiskit-aer
            pip install .

         if you're using pip >=19.0.0 then you can manually build a wheel file and install
         that instead.

         .. code:: sh

             cd qiskit-aer
             python setup.py bdist_wheel

         If you need to set a custom option during the wheel build you can refer to
         :ref:`aer_wheel_build_options`.

         After you build the python wheel it will be stored in the ``dist/`` dir in the
         Aer repository. The exact version will depend

         .. code:: sh

             cd dist
             pip install qiskit_aer-*.whl

         The exact filename of the output wheel file depends on the current version of
         Aer under development.

.. _aer_wheel_build_options:

Custom options during wheel builds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Aer build system uses `scikit-build <https://scikit-build.readthedocs.io/en/latest/index.html>`__
to run the compilation when building it with the python interface. It acts as an interface for
`setuptools <https://setuptools.readthedocs.io/en/latest/>`__ to call `CMake <https://cmake.org/>`__
and compile the binaries for your local system.

Due to the complexity of compiling the binaries you may need to pass options
to a certain part of the build process. The way to pass variables is:

.. code:: sh

   python setup.py bdist_wheel [skbuild_opts] [-- [cmake_opts] [-- build_tool_opts]]

where the elements within square brackets `[]` are optional, and
``skbuild_opts``, ``cmake_opts``, ``build_tool_opts`` are to be replaced by
flags of your choice. A list of *CMake* options is available here:
https://cmake.org/cmake/help/v3.6/manual/cmake.1.html#options. For
example, you could run something like:

.. code:: sh

   python setup.py bdist_wheel -- -- -j8

This is passing the flag `-j8` to the underlying build system (which in this
case is `Automake <https://www.gnu.org/software/automake/>`__) telling it that you want
to build in parallel using 8 processes.

For example, a common use case for these flags on linux is to specify a
specific version of the C++ compiler to use (normally if the default is too
old).

.. code:: sh

   python setup.py bdist_wheel -- -DCMAKE_CXX_COMPILER=g++-7

which will tell CMake to use the g++-7 command instead of the default g++ when
compiling Aer

Another common use case for this, depending on your environment, is that you may
need to specify your platform name and turn off static linking.

.. code:: sh

   python setup.py bdist_wheel --plat-name macosx-10.9-x86_64 \
   -- -DSTATIC_LINKING=False -- -j8

Here ``--plat-name`` is a flag to setuptools, to specify the platform name to
use in the package metadata, ``-DSTATIC_LINKING`` is a flag to CMake being used
to disable static linking, and ``-j8`` is a flag to Automake being used to use
8 processes for compilation.

A list of common options depending on platform are:

+--------+------------+----------------------+---------------------------------------------+
|Platform| Tool       | Option               | Use Case                                    |
+========+============+======================+=============================================+
| All    | Automake   | -j                   | Followed by a number this set the number of |
|        |            |                      | process to use for compilation              |
+--------+------------+----------------------+---------------------------------------------+
| Linux  | CMake      | -DCMAKE_CXX_COMPILER | Used to specify a specific C++ compiler,    |
|        |            |                      | this is often needed if you default g++ is  |
|        |            |                      | too.                                        |
+--------+------------+----------------------+---------------------------------------------+
| OSX    | setuptools | --plat-name          | Used to specify the platform name in the    |
|        |            |                      | output Python package.                      |
+--------+------------+----------------------+---------------------------------------------+
| OSX    | CMake      | -DSTATIC_LINKING     | Used to specify whether static linking      |
|        |            |                      | should be used or not                       |
+--------+------------+----------------------+---------------------------------------------+

.. note::
    Some of these options are not platform specific, if a platform is listed
    this is just outlining it's commonly used in that environment. Refer to the
    tool documentation for more information.


Installing Ignis from Source
----------------------------

1. Clone the ignis repository.

   .. code:: sh

      git clone https://github.com/Qiskit/qiskit-ignis.git

2. Cloning the repository creates a local directory called ``qiskit-ignis``.

   .. code:: sh

      cd qiskit-ignis

3. If you want to run tests or linting checks, install the developer requirements.
   This is not required to install or use the qiskit-ignis package when installing
   from source.

   .. code:: sh

      pip install -r requirements-dev.txt

4. Install ignis

   .. code:: sh

      pip install .

If you want to install it in editable mode, meaning that code changes to the
project don't require a reinstall to be applied you can do this with:

.. code:: sh

    pip install -e .

Installing Aqua from Source
---------------------------

1. Clone the Aqua repository.

   .. code:: sh

      git clone https://github.com/Qiskit/qiskit-aqua.git

2. Cloning the repository creates a local directory called ``qiskit-aqua``.

   .. code:: sh

      cd qiskit-aqua

3. If you want to run tests or linting checks, install the developer requirements.
   This is not required to install or use the qiskit-aqua package when installing
   from source.

   .. code:: sh

      pip install -r requirements-dev.txt

4. Install aqua

   .. code:: sh

      pip install .

If you want to install it in editable mode, meaning that code changes to the
project don't require a reinstall to be applied you can do this with:

.. code:: sh

    pip install -e .

Install Chemistry from Source
-----------------------------

1. Clone the qiskit-chemistry repository.

   .. code:: sh

     git clone https://github.com/Qiskit/qiskit-chemistry.git

2. Cloning the repository creates a local directory called ``qiskit-chemistry``.

   .. code:: sh

      cd qiskit-chemistry

3. If you want to run tests or linting checks, install the developer requirements.
   This is not required to install or use the qiskit-chemistry package when
   installing from source.

   .. code:: sh

      pip install -r requirements-dev.txt

4. Install qiskit-chemistry

   .. code:: sh

      pip install .

If you want to install it in editable mode, meaning that code changes to the
project don't require a reinstall to be applied you can do this with:

.. code:: sh

    pip install -e .
