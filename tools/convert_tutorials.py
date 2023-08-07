# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-function-docstring,broad-exception-caught

"""
Utility script to parallelise the conversion of several Jupyter notebooks.

If nbconvert starts offering built-in parallelisation this script can likely be dropped.
"""

import argparse
import multiprocessing
import os
import pathlib
import sys
import typing

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def worker(
    notebook_path: pathlib.Path, in_root: pathlib.Path, out_root: typing.Optional[pathlib.Path]
) -> typing.Optional[Exception]:
    """Single parallel worker that spawns a Jupyter executor node, executes the given notebook
    within it, and writes out the output."""
    try:
        print(f"({os.getpid()}) Processing '{str(notebook_path)}'", flush=True)
        processor = ExecutePreprocessor(timeout=300, kernel_name="python3")
        with open(notebook_path, "r") as fptr:
            notebook = nbformat.read(fptr, as_version=4)
        # Run the notebook with the working  directory set to the folder it resides in.
        processor.preprocess(notebook, {"metadata": {"path": f"{notebook_path.parent}/"}})

        # Ensure the output directory exists, and write to it.
        out_root = in_root if out_root is None else out_root
        out_path = out_root / notebook_path.relative_to(in_root).with_suffix(".nbconvert.ipynb")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fptr:
            nbformat.write(notebook, fptr)
    except Exception as exc:
        return exc
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute tutorial Jupyter notebooks.")
    parser.add_argument(
        "notebook_dirs", type=pathlib.Path, nargs="*", help="Folders containing Jupyter notebooks."
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        help="Output directory for files. Defaults to same location as input file.",
    )
    args = parser.parse_args()
    notebooks = sorted(
        {
            (notebook_path, in_root, args.out)
            for in_root in args.notebook_dirs
            for notebook_path in in_root.glob("**/*.ipynb")
        }
    )
    cpus = os.cpu_count()
    print(f"Using {cpus} processes.")
    with multiprocessing.Pool(cpus) as pool:
        failures = pool.starmap(worker, notebooks)
    num_failures = 0
    for path, failure in zip(notebooks, failures):
        if failure is not None:
            print(f"'{path}' failed: {failure}", file=sys.stderr)
            num_failures += 1
    return num_failures


if __name__ == "__main__":
    sys.exit(main())
