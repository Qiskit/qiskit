# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
.. _installing-nlopt:

Installing NLopt
================

In order to use any of the NLOpt based global optimizers NLopt must be installed.
The `NLopt download and installation instructions
<https://nlopt.readthedocs.io/en/latest/#download-and-installation>`__
describe how to do this.

If you running Aqua on Windows, then you might want to refer to the specific
`instructions for NLopt on Windows
<https://nlopt.readthedocs.io/en/latest/NLopt_on_Windows/>`__.

If you are running Aqua on a Unix-like system, first ensure that your environment is set
to the Python executable for which the Qiskit_Aqua package is installed and running.
Now, having downloaded and unpacked the NLopt archive file
(for example, ``nlopt-2.4.2.tar.gz`` for version 2.4.2), enter the following commands:

.. code:: sh

    ./configure --enable-shared --with-python
    make
    sudo make install

The above makes and installs the shared libraries and Python interface in `/usr/local`.
To have these be used by Aqua, the following commands can be entered to augment the dynamic
library load path and python path respectively, assuming that you choose to leave these
entities where they were built and installed as per above commands and that you
are running Python 3.6:

.. code:: sh

    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib64
    export PYTHONPATH=/usr/local/lib/python3.6/site-packages:${PYTHONPATH}

The two ``export`` commands above can be pasted into the ``.bash_profile`` file in the user's
home directory for automatic execution.  Now you can run Aqua and these optimizers should be
available for you to use.

"""
