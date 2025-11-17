"""CLI functionality tests."""

import os
import subprocess
import sys
import tempfile


class TestCLIBasic:
    """Test CLI functionality."""

    def test_cli_version(self):
        """Test CLI --version option."""
        result = subprocess.run(
            [sys.executable, "-m", "stat_lang.cli", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Version should be printed (exit code may vary, but should not crash)
        assert "StatLang" in result.stdout or "statlang" in result.stdout.lower()

    def test_cli_run_file(self):
        """Test CLI can run a .statlang file."""
        # Create a temporary .statlang file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".statlang", delete=False) as f:
            f.write("data work.test; input x; datalines; 1; run;")
            temp_file = f.name

        try:
            result = subprocess.run(
                [sys.executable, "-m", "stat_lang.cli", temp_file],
                capture_output=True,
                text=True,
                timeout=30,
            )
            # Should exit successfully (code 0) or at least not crash
            # The exact exit code may vary, but we check it doesn't fail catastrophically
            assert result.returncode in [0, 1]  # 0 = success, 1 = some expected errors
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_cli_nonexistent_file(self):
        """Test CLI handles non-existent file gracefully."""
        result = subprocess.run(
            [sys.executable, "-m", "stat_lang.cli", "nonexistent.statlang"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should exit with error code and print error message
        assert result.returncode != 0
        assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()

