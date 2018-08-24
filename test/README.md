# Qiskit-Terra Testing Framework

The purpose of this testing framework is to provide assurances regarding the correctness, quality and efficiency of Qiskit-Terra.


## Directory Structure
======================



## Running Tests
================

We use the  [standard Python "unittest" framework]
(<https://docs.python.org/3/library/unittest.html>) for our tests.

As our build system is based on CMake, we need to perform what is called an
"out-of-source" build before running the tests.
This is as simple as executing these commands:

Linux and Mac:

```shell
$ mkdir out
$ cd out
out$ cmake ..
out$ make
```

Windows:

```shell
C:\..\> mkdir out
C:\..\> cd out
C:\..\out> cmake -DUSER_LIB_PATH=C:\path\to\mingw64\lib\libpthreads.a -G "MinGW Makefiles" ..
C:\..\out> make
```


This will generate all needed binaries for your specific platform.

For executing the tests, a ``make test`` target is available.
The execution of the tests (both via the make target and during manual invocation)
takes into account the ``LOG_LEVEL`` environment variable. If present, a ``.log``
file will be created on the test directory with the output of the log calls, which
will also be printed to stdout. You can adjust the verbosity via the content
of that variable, for example:

Linux and Mac:

.. code-block:: bash

    $ cd out
    out$ LOG_LEVEL="DEBUG" ARGS="-V" make test

Windows:

.. code-block:: bash

    $ cd out
    C:\..\out> set LOG_LEVEL="DEBUG"
    C:\..\out> set ARGS="-V"
    C:\..\out> make test

For executing a simple python test manually, we don't need to change the directory
to ``out``, just run this command:


Linux and Mac:

.. code-block:: bash

    $ LOG_LEVEL=INFO python -m unittest test/python/test_apps.py

Windows:

.. code-block:: bash

    C:\..\> set LOG_LEVEL="INFO"
    C:\..\> python -m unittest test/python/test_apps.py

Testing options
===============

By default, and if there is no user credentials available, the tests that require online access are run with recorded (mocked) information. This is, the remote requests are replayed from a ``test/cassettes`` and not real HTTP requests is generated.
If user credentials are found, in that cases it use them to make the network requests.

How and which tests are executed is controlled by a environment variable ``QISKIT_TESTS``. The options are (where ``uc_available = True`` if the user credentials are available, and ``False`` otherwise): 

| Option | Description | Default | If `True`, forces
| --- | --- | --- | ----|
| `skip_online` | Skip tests that require remote requests (also, no mocked information is used). Does not require user credentials. | `False` | `rec = False`|
| `mock_online` | Run the online tests using mocked information. Does not require user credentials. | `not uc_available` | `skip_online = False`
| `run_slow` | Run tests tagged as slow (mostly those running on a device).  | `False` | |
`rec` | Record the remote requests. It requires user credentials. | `False` | `skip_online = False` `run_slow = False` |

It is possible to provide more than one option separated with commas.
The order of precedence in the options is right to left. For example, ``QISKIT_TESTS=skip_online,rec`` will set the options as ``skip_online == False`` and ``rec == True``.

