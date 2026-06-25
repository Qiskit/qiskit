# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for coverage.py configuration."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestCoverageConfig(unittest.TestCase):
    """Check that common non-executable patterns are excluded from reports."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[2]

    def test_exclude_also_patterns_are_loaded(self):
        """Project coverage settings should ignore overload stubs and ellipsis bodies."""
        result = subprocess.run(
            [sys.executable, "-m", "coverage", "debug", "config"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        config = result.stdout
        self.assertIn("@overload", config)
        self.assertIn("__main__", config)
        self.assertIn("\\.\\.\\.", config)

    def test_overload_stubs_are_not_reported_as_missing(self):
        module = """
from typing import overload

@overload
def pick(x: int) -> int:
    ...

@overload
def pick(x: str) -> str:
    ...

def pick(x):
    return x

if __name__ == "__main__":
    assert pick(1) == 1
"""
        with tempfile.TemporaryDirectory() as tmp:
            module_path = Path(tmp) / "overload_sample.py"
            module_path.write_text(module, encoding="utf-8")

            subprocess.run(
                [sys.executable, "-m", "coverage", "run", str(module_path)],
                cwd=self.repo_root,
                check=True,
                capture_output=True,
            )
            report = subprocess.run(
                [sys.executable, "-m", "coverage", "report", "-m", "--include", str(module_path)],
                cwd=self.repo_root,
                check=True,
                capture_output=True,
                text=True,
            )

        totals = [line for line in report.stdout.splitlines() if "overload_sample.py" in line]
        self.assertEqual(len(totals), 1)
        self.assertIn("100%", totals[0])


if __name__ == "__main__":
    unittest.main()
