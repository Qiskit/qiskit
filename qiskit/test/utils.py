# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utils for using with Qiskit unit tests."""

import logging
import os
from enum import Enum
from itertools import product

from qiskit import __path__ as qiskit_path


class Path(Enum):
    """Helper with paths commonly used during the tests."""

    # Main SDK path:    qiskit/
    SDK = qiskit_path[0]
    # test.python path: qiskit/test/python/
    TEST = os.path.normpath(os.path.join(SDK, "..", "test", "python"))
    # Examples path:    examples/
    EXAMPLES = os.path.normpath(os.path.join(SDK, "..", "examples"))
    # Schemas path:     qiskit/schemas
    SCHEMAS = os.path.normpath(os.path.join(SDK, "schemas"))
    # Sample QASMs path: qiskit/test/python/qasm
    QASMS = os.path.normpath(os.path.join(TEST, "qasm"))


def setup_test_logging(logger, log_level, filename):
    """Set logging to file and stdout for a logger.

    Args:
        logger (Logger): logger object to be updated.
        log_level (str): logging level.
        filename (str): name of the output file.
    """
    # Set up formatter.
    log_fmt = "{}.%(funcName)s:%(levelname)s:%(asctime)s:" " %(message)s".format(logger.name)
    formatter = logging.Formatter(log_fmt)

    # Set up the file handler.
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if os.getenv("STREAM_LOG"):
        # Set up the stream handler.
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # Set the logging level from the environment variable, defaulting
    # to INFO if it is not a valid level.
    level = logging._nameToLevel.get(log_level, logging.INFO)
    logger.setLevel(level)


class Case(dict):
    """<no description>"""

    pass


def generate_cases(docstring, dsc=None, name=None, **kwargs):
    """Combines kwargs in Cartesian product and creates Case with them"""
    ret = []
    keys = kwargs.keys()
    vals = kwargs.values()
    for values in product(*vals):
        case = Case(zip(keys, values))
        if docstring is not None:
            setattr(case, "__doc__", docstring.format(**case))
        if dsc is not None:
            setattr(case, "__doc__", dsc.format(**case))
        if name is not None:
            setattr(case, "__name__", name.format(**case))
        ret.append(case)
    return ret
