#!/usr/bin/env python3
"""
Unit tests for TimeFlies GUI functionality.
Tests the web-based GUI interface and command mapping.
"""

import subprocess
from unittest.mock import MagicMock, Mock, patch

import pytest

# Import the GUI module
from common.gui.gradio_launcher import TimeFliesWebGUI


class TestTimeFliesWebGUI:
    """Test cases for the TimeFlies web GUI."""

    def setup_method(self):
        """Setup test fixtures."""
        self.gui = TimeFliesWebGUI()

    def test_gui_initialization(self):
        """Test GUI initializes correctly."""
        assert self.gui.current_output == ""
        assert self.gui.is_running is False

    @patch("subprocess.run")
    def test_run_command_with_output_success(self, mock_run):
        """Test successful command execution."""
        # Mock successful subprocess result
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Success output"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        output, status = self.gui.run_command_with_output(
            ["echo", "test"], "Test Command"
        )

        assert "=== Test Command ===" in output
        assert "Success output" in output
        assert status == "✅ Success"
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_run_command_with_output_failure(self, mock_run):
        """Test failed command execution."""
        # Mock failed subprocess result
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error message"
        mock_run.return_value = mock_result

        output, status = self.gui.run_command_with_output(["false"], "Failed Command")

        assert "=== Failed Command ===" in output
        assert "Error message" in output
        assert status == "❌ Failed"

    @patch("subprocess.run")
    def test_run_command_timeout(self, mock_run):
        """Test command timeout handling."""
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 300)

        output, status = self.gui.run_command_with_output(
            ["sleep", "1000"], "Timeout Test"
        )

        assert "Command timed out" in output
        assert status == "❌ Timeout"

    @patch("subprocess.run")
    def test_check_installation_status_installed(self, mock_run):
        """Test installation status check when TimeFlies is installed."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "TimeFlies v1.0.0"
        mock_run.return_value = mock_result

        status = self.gui.check_installation_status()

        assert "✅ TimeFlies installed" in status
        assert "v1.0.0" in status
        mock_run.assert_called_with(
            ["timeflies", "--version"], capture_output=True, text=True, timeout=10
        )

    @patch("subprocess.run")
    def test_check_installation_status_not_installed(self, mock_run):
        """Test installation status check when TimeFlies is not installed."""
        mock_run.side_effect = FileNotFoundError()

        status = self.gui.check_installation_status()

        assert "❌ TimeFlies not found" in status

    def test_run_setup_command_construction(self):
        """Test setup command is constructed correctly."""
        with patch.object(self.gui, "run_command_with_output") as mock_run:
            mock_run.return_value = ("output", "status")

            # Mock progress object
            mock_progress = Mock()

            self.gui.run_setup(progress=mock_progress)

            mock_run.assert_called_with(["timeflies", "setup"], "TimeFlies Setup")
            mock_progress.assert_called()

    def test_run_train_command_construction(self):
        """Test training command is constructed with all options."""
        with patch.object(self.gui, "run_command_with_output") as mock_run:
            mock_run.return_value = ("output", "status")
            mock_progress = Mock()

            # Test with all options enabled
            self.gui.run_train(
                include_eda=True,
                include_analysis=True,
                batch_corrected=True,
                progress=mock_progress,
            )

            expected_cmd = [
                "timeflies",
                "train",
                "--with-eda",
                "--with-analysis",
                "--batch-corrected",
            ]
            mock_run.assert_called_with(expected_cmd, "TimeFlies Training")

            # Test with no options
            self.gui.run_train(
                include_eda=False,
                include_analysis=False,
                batch_corrected=False,
                progress=mock_progress,
            )

            expected_cmd_minimal = ["timeflies", "train"]
            mock_run.assert_called_with(expected_cmd_minimal, "TimeFlies Training")

    def test_run_batch_correction_command_construction(self):
        """Test batch correction command construction."""
        with patch.object(self.gui, "run_command_with_output") as mock_run:
            mock_run.return_value = ("output", "status")
            mock_progress = Mock()

            # Test Alzheimer's project
            self.gui.run_batch_correction(
                project="fruitfly_alzheimers", tissue="head", progress=mock_progress
            )

            expected_cmd = [
                "timeflies",
                "--alzheimers",
                "--tissue",
                "head",
                "batch-correct",
            ]
            mock_run.assert_called_with(expected_cmd, "Batch Correction")

            # Test aging project
            self.gui.run_batch_correction(
                project="fruitfly_aging", tissue="body", progress=mock_progress
            )

            expected_cmd = ["timeflies", "--aging", "--tissue", "body", "batch-correct"]
            mock_run.assert_called_with(expected_cmd, "Batch Correction")

    def test_run_evaluate_command_construction(self):
        """Test evaluation command construction."""
        with patch.object(self.gui, "run_command_with_output") as mock_run:
            mock_run.return_value = ("output", "status")
            mock_progress = Mock()

            self.gui.run_evaluate(
                include_eda=True,
                include_analysis=False,
                batch_corrected=True,
                progress=mock_progress,
            )

            expected_cmd = ["timeflies", "evaluate", "--with-eda", "--batch-corrected"]
            mock_run.assert_called_with(expected_cmd, "TimeFlies Evaluation")

    def test_run_queue_command_construction(self):
        """Test queue training command construction."""
        with patch.object(self.gui, "run_command_with_output") as mock_run:
            mock_run.return_value = ("output", "status")
            mock_progress = Mock()

            # Test with custom config and no resume
            self.gui.run_queue(
                config_file="custom_queue.yaml", no_resume=True, progress=mock_progress
            )

            expected_cmd = ["timeflies", "queue", "custom_queue.yaml", "--no-resume"]
            mock_run.assert_called_with(expected_cmd, "Model Queue Training")

            # Test with default config and resume
            self.gui.run_queue(config_file="", no_resume=False, progress=mock_progress)

            expected_cmd = ["timeflies", "queue"]
            mock_run.assert_called_with(expected_cmd, "Model Queue Training")

    @patch("pathlib.Path.exists")
    @patch("subprocess.run")
    def test_check_batch_environment_ready(self, mock_run, mock_exists):
        """Test batch environment check when ready."""
        mock_exists.return_value = True
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        status = self.gui.check_batch_environment()

        assert "✅ Batch environment ready" in status
        mock_run.assert_called_once()

    @patch("pathlib.Path.exists")
    def test_check_batch_environment_missing(self, mock_exists):
        """Test batch environment check when missing."""
        mock_exists.return_value = False

        status = self.gui.check_batch_environment()

        assert "❌ Batch environment not found" in status
        assert "timeflies setup --dev" in status

    def test_create_interface_returns_gradio_blocks(self):
        """Test that create_interface returns a Gradio Blocks object."""
        interface = self.gui.create_interface()

        # Check that it's a Gradio Blocks object
        import gradio as gr

        assert isinstance(interface, gr.Blocks)


class TestGUICommandCoverage:
    """Test that GUI covers all CLI commands."""

    def test_gui_has_all_major_cli_commands(self):
        """Test that GUI provides access to all major CLI functionality."""
        gui = TimeFliesWebGUI()

        # Check that GUI has methods for all major CLI commands
        assert hasattr(gui, "run_setup")  # timeflies setup
        assert hasattr(gui, "run_train")  # timeflies train
        assert hasattr(gui, "run_evaluate")  # timeflies evaluate
        assert hasattr(gui, "run_analyze")  # timeflies analyze
        assert hasattr(gui, "run_batch_correction")  # timeflies batch-correct
        assert hasattr(gui, "run_hyperparameter_tuning")  # timeflies tune
        assert hasattr(gui, "run_queue")  # timeflies queue
        assert hasattr(gui, "run_update")  # timeflies update
        assert hasattr(gui, "run_verify")  # timeflies verify

        # Check batch environment support
        assert hasattr(gui, "check_batch_environment")

        # Check status checking
        assert hasattr(gui, "check_installation_status")


class TestGUIIntegration:
    """Integration tests for GUI functionality."""

    @patch("common.gui.gradio_launcher.TimeFliesWebGUI")
    @patch("gradio.Blocks.launch")
    def test_launch_gui_function(self, mock_launch, mock_gui_class):
        """Test the launch_gui function."""
        from common.gui.gradio_launcher import launch_gui

        mock_gui_instance = Mock()
        mock_interface = Mock()
        mock_gui_instance.create_interface.return_value = mock_interface
        mock_gui_class.return_value = mock_gui_instance

        launch_gui(server_name="localhost", server_port=8080, share=True, debug=False)

        mock_gui_class.assert_called_once()
        mock_gui_instance.create_interface.assert_called_once()
        mock_interface.launch.assert_called_once_with(
            server_name="localhost",
            server_port=8080,
            share=True,
            debug=False,
            show_error=True,
            quiet=False,
        )


if __name__ == "__main__":
    pytest.main([__file__])
