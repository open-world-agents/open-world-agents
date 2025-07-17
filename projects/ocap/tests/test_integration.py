"""
Integration tests for the ocap command.

Tests that the ocap command runs without crashing during startup and initialization.
These tests verify the CLI interface works correctly without requiring full recording.
"""

import os
import subprocess
import tempfile
import time
from pathlib import Path

import pytest


def _get_subprocess_env():
    """Get environment variables for subprocess calls with proper encoding."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return env


class TestOcapIntegration:
    """Integration test for the ocap command - verifies it doesn't fail during startup."""

    def test_ocap_command_help_does_not_fail(self):
        """Test that 'ocap --help' command runs without errors."""
        try:
            # Test the help command which should not require any dependencies
            result = subprocess.run(
                ["ocap", "--help"],
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
                encoding="utf-8",
                env=_get_subprocess_env(),  # Ensure proper encoding in subprocess
            )

            # The command should succeed (exit code 0)
            assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

            # Should contain help text about recording
            assert "Record screen, keyboard, mouse" in result.stdout
            assert "file-location" in result.stdout or "FILE_LOCATION" in result.stdout

        except subprocess.TimeoutExpired:
            pytest.fail("Command timed out - this suggests a hanging process")

    def test_ocap_command_validation_does_not_fail(self):
        """Test that ocap command validates arguments without crashing."""
        temp_dir = tempfile.mkdtemp()
        try:
            test_file = Path(temp_dir) / "test-recording"
            process = None

            try:
                # Run ocap with a test file but immediately terminate
                # This tests initialization without actually recording
                process = subprocess.Popen(
                    ["ocap", str(test_file)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    env=_get_subprocess_env(),  # Ensure proper encoding in subprocess
                )

                # Give it a moment to initialize, then terminate
                time.sleep(2)
                process.terminate()

                # Wait for process to finish
                _, stderr = process.communicate(timeout=10)

                # Process should either:
                # 1. Exit cleanly (code 0) - if it started recording and was terminated
                # 2. Exit with specific error codes - but NOT crash with unhandled exceptions
                # 3. Be terminated (negative exit code on Unix, specific codes on Windows)

                # What we're checking: no unhandled Python exceptions in stderr
                # Common failure patterns to avoid:
                assert "Traceback" not in stderr, f"Unhandled exception occurred: {stderr}"
                assert "ImportError" not in stderr, f"Import error occurred: {stderr}"
                assert "ModuleNotFoundError" not in stderr, f"Module not found: {stderr}"

                # If it got far enough to start recording, that's success
                # (it would be terminated by our test, which is expected)

            except subprocess.TimeoutExpired:
                if process:
                    process.kill()
                pytest.fail("Command initialization took too long")
            finally:
                # Ensure process is cleaned up
                if process and process.poll() is None:
                    process.kill()
                    process.wait()

                # Give a moment for file handles to be released
                time.sleep(0.5)

        finally:
            # Manual cleanup with retry for Windows file locking issues
            import shutil

            for attempt in range(3):
                try:
                    shutil.rmtree(temp_dir)
                    break
                except PermissionError:
                    if attempt < 2:
                        time.sleep(1)  # Wait and retry
                    else:
                        # If we still can't delete, that's okay for this test
                        # The important thing is that ocap didn't crash
                        pass

    def test_ocap_command_with_invalid_args_fails_gracefully(self):
        """Test that ocap command fails gracefully with invalid arguments."""
        try:
            # Test with invalid arguments - should fail but not crash
            result = subprocess.run(
                ["ocap", "--invalid-flag"],
                capture_output=True,
                text=True,
                timeout=10,
                encoding="utf-8",
                env=_get_subprocess_env(),  # Ensure proper encoding in subprocess
            )

            # Should fail with non-zero exit code
            assert result.returncode != 0

            # Should not have unhandled exceptions
            assert "Traceback" not in result.stderr
            assert "ImportError" not in result.stderr
            assert "ModuleNotFoundError" not in result.stderr

        except subprocess.TimeoutExpired:
            pytest.fail("Command with invalid args timed out")
