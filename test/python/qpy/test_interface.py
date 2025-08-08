import unittest
from qiskit.qpy import load
from qiskit.exceptions import QiskitError
from qiskit.user_config import get_config
import io

class TestQpyLoad(unittest.TestCase):
    def setUp(self):
        self.config = get_config()

    def test_enforce_min_qpy_version(self):
        """Test that QPY file below min_qpy_version raises QiskitError."""
        file_obj = io.BytesIO(b"QPY_VERSION=5")
        self.config.settings["min_qpy_version"] = 10
        with self.assertRaises(QiskitError) as cm:
            load(file_obj)
        self.assertIn("below the minimum version 10", str(cm.exception))

    def test_skip_enforcement_if_unset(self):
        """Test that load proceeds if min_qpy_version is unset."""
        file_obj = io.BytesIO(b"QPY_VERSION=5")
        self.config.settings.pop("min_qpy_version", None)
        load(file_obj)  # Mocked to not fail on other logic

    def test_valid_qpy_version(self):
        """Test that QPY file meeting min_qpy_version loads successfully."""
        file_obj = io.BytesIO(b"QPY_VERSION=10")
        self.config.settings["min_qpy_version"] = 10
        load(file_obj)  # Mocked to not fail on other logic

    def test_invalid_qpy_version_type(self):
        """Test that non-integer QPY version raises QiskitError."""
        file_obj = io.BytesIO(b"QPY_VERSION=invalid")
        self.config.settings["min_qpy_version"] = 10
        with self.assertRaises(QiskitError) as cm:
            load(file_obj)
        self.assertIn("not an integer", str(cm.exception))

    def test_invalid_min_qpy_version_type(self):
        """Test that invalid min_qpy_version type raises QiskitError."""
        file_obj = io.BytesIO(b"QPY_VERSION=5")
        self.config.settings["min_qpy_version"] = "invalid"
        with self.assertRaises(QiskitError) as cm:
            load(file_obj)
        self.assertIn("must be a positive integer", str(cm.exception))
