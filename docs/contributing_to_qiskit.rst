
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

Installing Terra from Source
----------------------------
Installing from source requires that you have a c++ compiler on your system that supports
c++-11.  On most Linux platforms, the necessary GCC compiler is already installed.

Installing a Compiler for macOS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you use macOS, you can install the Clang compiler by installing XCode.
Check if you have XCode and clang installed by opening a terminal window and entering the
following.

.. code:: sh

  clang --version

Install XCode and clang by using the following command.

.. code:: sh

    xcode-select --install

Installing a Compiler for Windows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
On Windows, it is easiest to install the Visual C++ compiler from the
`Build Tools for Visual Studio 2017 <https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017>`_.
You can instead install Visual Studio version 2015 or 2017, making sure to select the
options for installing the C++ compiler.

Installing Qiskit Terra
^^^^^^^^^^^^^^^^^^^^^^^
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

4. Install ``qiskit-terra``.

.. code:: sh

    pip install .

* If you want to install it in editable mode, meaning that code changes to the
  project don't require a reinstall to be applied you can do this with:

.. code:: sh

    pip install -e .

* You can then run the code examples working after installing terra. You can
  run the example with the following command.

.. code:: sh

    python examples/python/using_qiskit_terra_level_0.py


After you've installed Terra, you can install Aer as an add-on to run additional simulators.

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

Linux
^^^^^

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

macOS
^^^^^

3. Install dependencies.

To use the `Clang`_ compiler on macOS, you need to install an extra library for
supporting `OpenMP`_.  You can use `brew`_ to install this and other
dependencies.

.. _brew: https://brew.sh/
.. _Clang: https://clang.llvm.org/
.. _OpenMP: https://www.openmp.org/

.. code:: sh

    brew install libomp

You then also have to install a BLAS implementation, `OpenBLAS`_ is the
default choice.

.. code:: sh

    brew install openblas

.. _OpenBlas: https://www.openblas.net/

You also need to have ``Xcode Command Line Tools`` installed.

.. code:: sh

    xcode-select --install

4. Build and install qiskit

If you have pip <19.0.0 installed and your environment doesn't require a
custom build options you can just run:

.. code:: sh

    cd qiskit-aer
    pip install .

This will both build the binaries and install aer.

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

Windows
^^^^^^^

On Windows you need to use `Anaconda3`_ or `Miniconda3`_ to install all the
dependencies.

.. _Anaconda3: https://www.anaconda.com/distribution/#windows
.. _Miniconda3: https://docs.conda.io/en/latest/miniconda.html

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Aer build system uses `scikit-build`_ to run the compilation when building
it with the python interface. It acts as an interface for `setuptools`_ to
call `CMake`_ and compile the binaries for your local system.

.. _scikit-build: https://scikit-build.readthedocs.io/en/latest/index.html
.. _setuptools: https://setuptools.readthedocs.io/en/latest/
.. _CMake: https://cmake.org/

Due to the complexity of compiling the binaries you may need to pass options
to a certain part of the build process. The way to pass variables is:

.. code:: sh

    python setup.py bdist_wheel [skbuild_opts] [-- [cmake_opts] [-- build_tool_opts]]

where the elements within square brackets `[]` are optional, and
``skbuild_opts``, ``cmake_opts``, ``build_tool_opts`` are to be replaced by
flags of your choice. A list of *CMake* options is available here:
https://cmake.org/cmake/help/v3.6/manual/cmake.1.html#options . For
example, you could run something like:

.. code:: sh

    python setup.py bdist_wheel -- -- -j8

This is passing the flag `-j8` to the underlying build system (which in this
case is `Automake`_) telling it that you want to build in parallel using 8
processes.

.. _Automake: https://www.gnu.org/software/automake/

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

* If you want to install it in editable mode, meaning that code changes to the
  project don't require a reinstall to be applied you can do this with:

.. code:: sh

    pip install -e .


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

* If you want to install it in editable mode, meaning that code changes to the
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

... code:: sh

    pip install -r requirements-dev.txt

4. Install aqua

.. code:: sh

    pip install .

* If you want to install it in editable mode, meaning that code changes to the
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

* If you want to install it in editable mode, meaning that code changes to the
  project don't require a reinstall to be applied you can do this with:

.. code:: sh

    pip install -e .
