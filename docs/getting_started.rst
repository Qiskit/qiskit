:orphan:

###############
Getting started
###############

.. _installation:

Installation
============

Let's get started using Qiskit!  The first thing to do is choose how you're
going to run and install the packages.  There are three main ways to do this:

.. tab-set::

    .. tab-item:: Start locally

        Qiskit supports Python 3.7 or later. However, both Python and Qiskit are
        evolving ecosystems, and sometimes when new releases occur in one or the other,
        there can be problems with compatibility.

        You will need to `download Python <https://wiki.python.org/moin/BeginnersGuide/Download>`__
        on your local system to get started. `Jupyter <https://jupyter.org/install>`__ is recommended for
        interacting with Qiskit.

        We recommend using `Python virtual environments <https://docs.python.org/3.10/tutorial/venv.html>`__
        to cleanly separate Qiskit from other applications and improve your experience.

        Create a minimal environment with only Python installed in it.

        .. code:: text

            python3 -m venv /path/to/virtual/environment

        Activate your new environment.

        .. code:: text

            source /path/to/virtual/environment/bin/activate


        Note: if you are using Windows, use the following commands in PowerShell.

        .. code:: text

           python3 -m venv c:\path\to\virtual\environment
           c:\path\to\virtual\environment\Scripts\Activate.ps1


        Next, install the Qiskit package.

        .. code:: text

            pip install qiskit

        If the packages were installed correctly, you can run ``pip list`` to see the active
        packages in your virtual environment.

        If you intend to use visualization functionality or Jupyter notebooks it is
        recommended to install Qiskit with the extra ``visualization`` support:

        .. code:: text

            pip install qiskit[visualization]

        It is worth pointing out that if you're a zsh user (which is the default shell on newer
        versions of macOS), you'll need to put ``qiskit[visualization]`` in quotes:

        .. code:: text

            pip install 'qiskit[visualization]'

    .. tab-item:: Start on the cloud

        The following cloud vendors have Qiskit pre-installed in their environments:

       .. qiskit-card::
          :header: IBM Quantum Lab
          :card_description: Build quantum applications and experiments with Qiskit in a cloud programming environment.
          :image: _static/images/ibm_qlab.png
          :link: https://quantum-computing.ibm.com/

       .. qiskit-card::
          :header: Strangeworks
          :card_description: A platform that enables users and organizations to easily apply quantum computing to their most pressing problems and research.
          :image: _static/images/strangeworks.png
          :link: https://strangeworks.com/

    .. tab-item:: Install from source

       Installing Qiskit from source allows you to access the current development
       version, instead of using the version in the Python Package
       Index (PyPI) repository. This will give you the ability to inspect and extend
       the latest version of the Qiskit code more efficiently.

       Begin by making a new virtual environment and activating it:

       .. code-block:: text

          python3 -m venv QiskitDevenv
          source QiskitDevenv/bin/activate

       Installing from source requires that you have the Rust compiler on your system.
       To install the Rust compiler the recommended path is to use rustup, which is
       a cross-platform Rust installer. To use rustup you can go to:

       https://rustup.rs/

       which will provide instructions for how to install rust on your platform.
       Besides rustup there are
       `other installation methods <https://forge.rust-lang.org/infra/other-installation-methods.html>`__ available too.

       Once the Rust compiler is installed, you are ready to install Qiskit.

       1. Clone the Qiskit repository.

          .. code:: text

             git clone https://github.com/Qiskit/qiskit-terra.git

       2. Cloning the repository creates a local folder called ``qiskit-terra``.

          .. code:: text

             cd qiskit-terra

       3. If you want to run tests or linting checks, install the developer requirements.

          .. code:: text

             pip install -r requirements-dev.txt

       4. Install ``qiskit-terra``.

          .. code:: text

             pip install .

       If you want to install it in editable mode, meaning that code changes to the
       project don't require a reinstall to be applied, you can do this with:

       .. code:: text

          pip install -e .

       Installing in editable mode will build the compiled extensions in debug mode
       without optimizations. This will affect the runtime performance of the compiled
       code. If you'd like to use editable mode and build the compiled code in release
       with optimizations enabled you can run:

       .. code:: text

           python setup.py build_rust --release --inplace

       after you run pip and that will rebuild the binary in release mode.
       If you are working on Rust code in Qiskit you will need to rebuild the extension
       code every time you make a local change. ``pip install -e .`` will only build
       the Rust extension when it's called, so any local changes you make to the Rust
       code after running pip will not be reflected in the installed package unless
       you rebuild the extension. You can leverage the above ``build_rust`` command to
       do this (with or without ``--release`` based on whether you want to build in
       debug mode or release mode).

       You can then run the code examples after installing Qiskit. You can
       run the example with the following command.

       .. code:: text

          python examples/python/using_qiskit_terra_level_0.py

.. _platform_support:

Platform Support
----------------

Qiskit strives to support as many platforms as possible, but due to limitations
in available testing resources and platform availability, not all platforms
can be supported. Platform support for Qiskit is broken into 3 tiers with different
levels of support for each tier. For platforms outside these, Qiskit is probably
still installable, but it's not tested and you will have to build Qiskit (and likely
Qiskit's dependencies) from source.

Additionally, Qiskit only supports CPython. Running with other Python
interpreters isn't currently supported.

Tier 1
''''''

Tier 1 supported platforms are fully tested upstream as part of the development
processes to ensure any proposed change will function correctly. Pre-compiled
binaries are built, tested, and published to PyPI as part of the release process.
These platforms are expected to be installable with just a functioning Python
environment as all dependencies are available on these platforms.

Tier 1 platforms are currently:

 * Linux x86_64 (distributions compatible with the
   `manylinux 2014 <https://www.python.org/dev/peps/pep-0599/>`__
   packaging specification).
 * macOS x86_64 (10.9 or newer)
 * Windows 64 bit

Tier 2
''''''

Tier 2 platforms are not tested upstream as part of development process. However,
pre-compiled binaries are built, tested, and published to PyPI as part of the
release process and these packages can be expected to be installed with just a
functioning Python environment.

Tier 2 platforms are currently:

 * Linux i686 (distributions compatible with the
   `manylinux 2014 <https://www.python.org/dev/peps/pep-0599/>`__ packaging
   specification) for Python < 3.10
 * Windows 32 bit for Python < 3.10
 * Linux aarch64 (distributions compatible with the
   `manylinux 2014 <https://www.python.org/dev/peps/pep-0599/>`__ packaging
   specification)

Tier 3
''''''

Tier 3 platforms are not tested upstream as part of the development process.  Pre-compiled
binaries are built and published to PyPI as part of the release process, with no
testing at all. They may not be installable with just a functioning Python
environment and may require a C/C++ compiler or additional programs to build
dependencies from source as part of the installation process. Support for these
platforms are best effort only.

Tier 3 platforms are currently:

 * Linux ppc64le (distributions compatible with the
   `manylinux 2014 <https://www.python.org/dev/peps/pep-0599/>`__ packaging
   specification)
 * Linux s390x (distributions compatible with the
   `manylinux 2014 <https://www.python.org/dev/peps/pep-0599/>`__ packaging
   specification)
 * macOS arm64 (10.15 or newer)
 * Linux i686 (distributions compatible with the
   `manylinux 2014 <https://www.python.org/dev/peps/pep-0599/>`__ packaging
   specification) for Python >= 3.10
 * Windows 32 bit for Python >= 3.10

Ready to get going?...
======================

.. qiskit-call-to-action-grid::

   .. qiskit-call-to-action-item::
      :description: Learn how to build, execute, and post-process quantum circuits with Qiskit.
      :header: Qiskit from the ground up
      :button_link:  intro_tutorial1.html
      :button_text: Start learning Qiskit

   .. qiskit-call-to-action-item::
      :description: Find out how to leverage Qiskit for everything from single-circuits to full quantum application development.
      :header: Dive into the tutorials
      :button_link:  tutorials.html
      :button_text: Qiskit tutorials


.. Hiding - Indices and tables
   :ref:`genindex`
   :ref:`modindex`
   :ref:`search`
