# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Utilities for logging."""

import logging
from logging.config import dictConfig


class SimpleInfoFormatter(logging.Formatter):
    """Custom Formatter that uses a simple format for INFO."""
    _style_info = logging._STYLES['%'][0]('%(message)s')

    def formatMessage(self, record):
        if record.levelno == logging.INFO:
            return self._style_info.format(record)
        return logging.Formatter.formatMessage(self, record)


QISKIT_LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'f': {
            '()': SimpleInfoFormatter,
            'format': '%(asctime)s:%(name)s:%(levelname)s: %(message)s'
        },
    },
    'handlers': {
        'h': {
            'class': 'logging.StreamHandler',
            'formatter': 'f'
        }
    },
    'loggers': {
        'qiskit': {
            'handlers': ['h'],
            'level': logging.INFO,
        },
    }
}


def set_qiskit_logger():
    """Update 'qiskit' logger configuration using a SDK default one.

    Update the configuration of the 'qiskit' logger using the default SDK
    configuration provided by `QISKIT_LOGGING_CONFIG`:

    * console logging using a custom format for levels != INFO.
    * console logging with simple format for level INFO.
    * set logger level to INFO.

    Warning:
        This function modifies the configuration of the standard logging system
        for the 'qiskit.*' loggers, and might interfere with custom logger
        configurations.
    """
    dictConfig(QISKIT_LOGGING_CONFIG)


def unset_qiskit_logger():
    """Remove the handlers for the 'qiskit' logger."""
    qiskit_logger = logging.getLogger('qiskit')
    for handler in qiskit_logger.handlers:
        qiskit_logger.removeHandler(handler)
