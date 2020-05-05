#!/usr/bin/env python3

# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.




import argparse
from datetime import datetime
import json
import os

from qiskit import IBMQ
from qiskit.circuit.parameterexpression import ParameterExpression


class BackendEncoder(json.JSONEncoder):
    """A json encoder for qobj"""

    def default(self, o):
        # Convert numpy arrays:
        if hasattr(o, 'tolist'):
            return o.tolist()
        # Use Qobj complex json format:
        if isinstance(o, complex):
            return [o.real, o.imag]
        if isinstance(o, ParameterExpression):
            return float(o)
        if isinstance(o, datetime):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)


DEFAULT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'qiskit', 'test', 'mock', 'backends')


def main():
    parser = argparse.ArgumentParser(
        description='Generate fake backend snapshots')
    parser.add_argument('--dir', '-d', type=str, default=DEFAULT_DIR)
    parser.add_argument('backends', type=str, nargs='*')
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--hub', type=str, default=None)
    parser.add_argument('--group', type=str, default=None)
    args = parser.parse_args()
    provider = IBMQ.load_account()
    if args.hub or args.group or args.project:
        provider = IBMQ.get_provider(hub=args.hub, group=args.group,
                                     project=args.project)
    ibmq_backends = provider.backends()
    for backend in ibmq_backends:
        raw_name = backend.name()
        if 'sim' in raw_name:
            continue
        if raw_name == 'ibmqx2':
            name = 'yorktown'
        else:
            name = raw_name.split('_')[1]
            if name == '16':
                name = 'melbourne'
        if not args.backends or (name in args.backends or
                                 raw_name in args.backends):
            if not os.path.isdir(os.path.join(args.dir, name)):
                print("Skipping, fake backend for %s does not exist yet" % name)
                continue
            config = backend.configuration()
            props = backend.properties()
            defs = backend.defaults()
            if config:
                config_path = os.path.join(args.dir, name,
                                           'conf_%s.json' % name)
                config_dict = config.to_dict()
                if 'dt' in config_dict:
                    config_dict['dt'] /= 1e-9
                if 'dtm' in config_dict:
                    config_dict['dtm'] /= 1e-9
                if 'qubit_lo_range'in config_dict:
                    config_dict['qubit_lo_range'] = [
                        [min_range / 1e9, max_range / 1e9] for
                        (min_range,
                         max_range) in config_dict['qubit_lo_range']]
                if 'meas_lo_range' in config_dict:
                    config_dict['meas_lo_range'] = [
                        [min_range / 1e9, max_range / 1e9] for
                        (min_range, max_range) in config_dict['meas_lo_range']]
                if 'rep_times' in config_dict:
                    config_dict['rep_times'] = [
                        _rt / 1e-6 for _rt in config_dict['rep_times']]

                with open(config_path, 'w') as fd:
                    fd.write(json.dumps(config_dict,
                                        cls=BackendEncoder))
            if props:
                props_path = os.path.join(args.dir, name,
                                          'props_%s.json' % name)
                with open(props_path, 'w') as fd:
                    fd.write(json.dumps(props.to_dict(),
                                        cls=BackendEncoder))
            if defs:
                defs_path = os.path.join(args.dir, name,
                                         'defs_%s.json' % name)
                with open(defs_path, 'w') as fd:
                    fd.write(json.dumps(defs.to_dict(),
                                        cls=BackendEncoder))


if __name__ == main():
    main()
