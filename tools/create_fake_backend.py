#!/usr/bin/env python3

# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility script to create fake backends"""

import argparse
import sys
from string import Template
import os
from reno.utils import get_random_string

from update_fake_backends import DEFAULT_DIR
from qiskit import IBMQ

RENO_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "releasenotes",
    "notes",
)

HEADER = """# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""

init_py = Template(
    """${HEADER}
\"\"\"Fake ${capital_backend_name} device (${no_qubits} qubits)\"\"\"

from .fake_${backend_name} import Fake${capital_backend_name}
"""
)

fake_be_py = Template(
    """${HEADER}

\"\"\"
Fake ${capital_backend_name} device (${no_qubits} qubits).
\"\"\"

import os
from qiskit.providers.fake_provider import fake_backend


class Fake${capital_backend_name}(fake_backend.FakeBackendV2):
    \"\"\"A fake ${no_qubits} qubit backend.\"\"\"

    dirname = os.path.dirname(__file__)
    conf_filename = "conf_${backend_name}.json"
    props_filename = "props_${backend_name}.json"
    defs_filename = "defs_${backend_name}.json"
    backend_name = "fake_${backend_name}"

"""
)

reno_template = Template(
    """---
features:
  - |
    The fake backend :class:`~Fake${capital_backend_name}` was added with the information
    from IBM Quantum `${system_name}` system.
"""
)


def _insert_line_in_section(
    line_to_insert, section, line_in_section_starts_with, backend_dir, file_to_modify
):
    tmp_file = os.path.join(backend_dir, "_tmp_.py")
    with open(file_to_modify, "r") as original, open(tmp_file, "a+") as destination:
        previous_line = ""
        in_section = False
        for line in original.readlines():
            if line.startswith(section):
                in_section = True
            elif (
                in_section
                and line.startswith(line_in_section_starts_with)
                and previous_line < line_to_insert < line
            ):
                in_section = False
                destination.write(line_to_insert)
            destination.write(line)
    os.replace(tmp_file, file_to_modify)


def _main():
    parser = argparse.ArgumentParser(description="Create fake backend")
    parser.add_argument("--dir", "-d", type=str, default=DEFAULT_DIR)
    parser.add_argument("backend_name", type=str, default=None)
    parser.add_argument("system_name", type=str, default=None)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--hub", type=str, default=None)
    parser.add_argument("--group", type=str, default=None)
    args = parser.parse_args()

    backend_dir = os.path.join(args.dir, args.backend_name)
    if os.path.isdir(backend_dir):
        print(f"ERROR: Directory {backend_dir} already exists")
        sys.exit(1)

    provider = IBMQ.load_account()
    if args.hub or args.group or args.project:
        provider = IBMQ.get_provider(hub=args.hub, group=args.group, project=args.project)
    ibmq_backend = provider.get_backend(args.system_name)

    vars_ = {
        "HEADER": HEADER,
        "backend_name": args.backend_name,
        "capital_backend_name": args.backend_name.capitalize(),
        "no_qubits": len(ibmq_backend.properties().qubits),
        "system_name": args.system_name,
    }

    # Step 1. Create the directory for the backend <backend_dir>
    os.mkdir(backend_dir)

    # Step 2. <backend_dir>/__init__.py
    result = init_py.substitute(vars_)
    with open(os.path.join(backend_dir, "__init__.py"), "w") as fd:
        fd.write(result)

    # Step 3. <backend_dir>/fake_<backend_name>.py
    result = fake_be_py.substitute(vars_)
    with open(os.path.join(backend_dir, f"fake_{args.backend_name}.py"), "w") as fd:
        fd.write(result)

    # Step 4. update <backend_dir>/../__init__.py
    init_file = os.path.join(backend_dir, "..", "__init__.py")
    backend_v2_line = f"from .{vars_['backend_name']} import Fake{vars_['capital_backend_name']}\n"
    _insert_line_in_section(backend_v2_line, "# BackendV2", "from", backend_dir, init_file)

    # Step 5. update <backend_dir>/../../__init__.py
    init_file = os.path.join(backend_dir, "..", "..", "__init__.py")
    backend_v2_line = f"    Fake{vars_['capital_backend_name']}\n"
    _insert_line_in_section(backend_v2_line, "Fake V2 Backends", "    ", backend_dir, init_file)

    # Step 6. update <backend_dir>/../../fake_provider.py
    init_file = os.path.join(backend_dir, "..", "..", "fake_provider.py")
    backend_v2_line = f"            Fake{vars_['capital_backend_name']}(),\n"
    _insert_line_in_section(
        backend_v2_line,
        "class FakeProviderForBackend",
        "            Fake",
        backend_dir,
        init_file,
    )

    # Step 7. releasenotes/notes/fake_<backend_name>-<random_string>.yaml
    result = reno_template.substitute(vars_)
    with open(
        os.path.join(RENO_DIR, f"fake_{args.backend_name}-{get_random_string()}.yaml"), "w"
    ) as fd:
        fd.write(result)


if __name__ == "__main__":
    _main()
