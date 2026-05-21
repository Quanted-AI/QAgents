"""Tests for wheel build verification.

Validates that the built .whl file contains only the expected package
files and dist-info metadata, that the wheel version matches
quanted_agents.__version__, and that the wheel can be installed via
pip dry-run.
"""

import glob
import os
import shutil
import subprocess
import tempfile
import unittest
import zipfile

import quanted_agents


@unittest.skipUnless(shutil.which("uv"), "uv build tool not installed")
class TestWheelBuild(unittest.TestCase):
    """Verify wheel contents, version, and installability."""

    tmp_dir: str
    whl_path: str

    def setUp(self) -> None:
        """Build wheel into a temporary directory for inspection."""
        self.tmp_dir = tempfile.mkdtemp(prefix="wheel_test_")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", self.tmp_dir],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            self.fail(f"Wheel build failed: {result.stderr}")
        wheels = glob.glob(os.path.join(self.tmp_dir, "*.whl"))
        if not wheels:
            self.fail(f"No .whl file found in {self.tmp_dir}")
        self.whl_path = wheels[0]

    def tearDown(self) -> None:
        """Remove temporary build directory."""
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_wheel_contains_only_package_code(self) -> None:
        """Verify wheel contains only quanted_agents/ and dist-info entries.

        Ensures no test files, examples, docs, or planning files leak
        into the distribution wheel. Also confirms py.typed marker and
        workflows subpackage are present.
        """
        with zipfile.ZipFile(self.whl_path, "r") as zf:
            names = zf.namelist()

        # Every entry must start with quanted_agents/ or contain dist-info
        for name in names:
            is_package = name.startswith("quanted_agents/")
            is_dist_info = "dist-info" in name
            self.assertTrue(
                is_package or is_dist_info,
                f"Unexpected file in wheel: {name}",
            )

        # Forbidden content patterns (case-insensitive)
        forbidden_patterns = ["test", "example", ".planning", "docs/"]
        names_lower = [n.lower() for n in names]
        for pattern in forbidden_patterns:
            for name_lower in names_lower:
                self.assertNotIn(
                    pattern,
                    name_lower,
                    f"Forbidden pattern '{pattern}' found in wheel entry: {name_lower}",
                )

        # py.typed marker must be present
        self.assertIn("quanted_agents/py.typed", names, "py.typed marker missing from wheel")

        # workflows subpackage must be present
        self.assertIn(
            "quanted_agents/workflows/__init__.py",
            names,
            "workflows subpackage __init__.py missing from wheel",
        )

    def test_wheel_version_matches_init(self) -> None:
        """Verify wheel filename version matches quanted_agents.__version__.

        The wheel filename follows the pattern:
        quanted_agents-{version}-py3-none-any.whl
        """
        filename = os.path.basename(self.whl_path)
        # Pattern: quanted_agents-{version}-py3-none-any.whl
        parts = filename.split("-")
        wheel_version = parts[1]
        self.assertEqual(
            wheel_version,
            quanted_agents.__version__,
            f"Wheel version '{wheel_version}' does not match "
            f"__init__.py version '{quanted_agents.__version__}'",
        )

    def test_wheel_installs_dry_run(self) -> None:
        """Verify wheel can be installed via pip dry-run without network access."""
        result = subprocess.run(
            ["pip", "install", "--no-deps", "--dry-run", self.whl_path],
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"Dry-run install failed: {result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
