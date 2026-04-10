#!/usr/bin/env python3
"""
TimeFlies Gradio Web GUI
A modern web-based interface for TimeFlies installation and management.

This replaces the tkinter GUI with a web interface that works everywhere.
"""

import os
import subprocess
import sys
import threading
from pathlib import Path

import gradio as gr


class TimeFliesWebGUI:
    def __init__(self):
        self.current_output = ""
        self.is_running = False

    def run_command_with_output(
        self, cmd: list, description: str = ""
    ) -> tuple[str, str]:
        """Run a command and return output and status."""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            output = f"=== {description} ===\n"
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n"
            output += f"Return code: {result.returncode}\n\n"

            status = "✅ Success" if result.returncode == 0 else "❌ Failed"
            return output, status

        except subprocess.TimeoutExpired:
            return f"❌ Command timed out: {' '.join(cmd)}\n\n", "❌ Timeout"
        except Exception as e:
            return f"❌ Error running command: {e}\n\n", "❌ Error"

    def check_installation_status(self) -> str:
        """Check current TimeFlies installation status."""
        try:
            result = subprocess.run(
                ["timeflies", "--version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return f"✅ TimeFlies installed: {result.stdout.strip()}"
            else:
                return "❌ TimeFlies not installed or not accessible"
        except Exception:
            return "❌ TimeFlies not found"

    def run_setup(self, progress=gr.Progress()) -> tuple[str, str]:
        """Run TimeFlies setup."""
        progress(0.1, "Starting setup...")
        output, status = self.run_command_with_output(
            ["timeflies", "setup"], "TimeFlies Setup"
        )
        progress(1.0, "Setup complete")
        return output, status

    def run_train(
        self,
        include_eda: bool,
        include_analysis: bool,
        batch_corrected: bool,
        progress=gr.Progress(),
    ) -> tuple[str, str]:
        """Run TimeFlies training."""
        cmd = ["timeflies", "train"]
        if include_eda:
            cmd.append("--with-eda")
        if include_analysis:
            cmd.append("--with-analysis")
        if batch_corrected:
            cmd.append("--batch-corrected")

        progress(0.1, "Starting training...")
        output, status = self.run_command_with_output(cmd, "TimeFlies Training")
        progress(1.0, "Training complete")
        return output, status

    def run_batch_correction(
        self, project: str, tissue: str, progress=gr.Progress()
    ) -> tuple[str, str]:
        """Run batch correction."""
        project_flag = "--alzheimers" if project == "fruitfly_alzheimers" else "--aging"
        cmd = ["timeflies", project_flag, "--tissue", tissue, "batch-correct"]

        progress(0.1, "Starting batch correction...")
        output, status = self.run_command_with_output(cmd, "Batch Correction")
        progress(1.0, "Batch correction complete")
        return output, status

    def run_hyperparameter_tuning(self, progress=gr.Progress()) -> tuple[str, str]:
        """Run hyperparameter tuning."""
        progress(0.1, "Starting hyperparameter tuning...")
        output, status = self.run_command_with_output(
            ["timeflies", "tune"], "Hyperparameter Tuning"
        )
        progress(1.0, "Tuning complete")
        return output, status

    def run_update(self, progress=gr.Progress()) -> tuple[str, str]:
        """Update TimeFlies."""
        progress(0.1, "Updating TimeFlies...")
        output, status = self.run_command_with_output(
            ["timeflies", "update"], "TimeFlies Update"
        )
        progress(1.0, "Update complete")
        return output, status

    def run_evaluate(
        self,
        include_eda: bool,
        include_analysis: bool,
        batch_corrected: bool,
        progress=gr.Progress(),
    ) -> tuple[str, str]:
        """Run TimeFlies evaluation."""
        cmd = ["timeflies", "evaluate"]
        if include_eda:
            cmd.append("--with-eda")
        if include_analysis:
            cmd.append("--with-analysis")
        if batch_corrected:
            cmd.append("--batch-corrected")

        progress(0.1, "Starting evaluation...")
        output, status = self.run_command_with_output(cmd, "TimeFlies Evaluation")
        progress(1.0, "Evaluation complete")
        return output, status

    def run_analyze(
        self, batch_corrected: bool, progress=gr.Progress()
    ) -> tuple[str, str]:
        """Run project analysis."""
        cmd = ["timeflies", "analyze"]
        if batch_corrected:
            cmd.append("--batch-corrected")

        progress(0.1, "Starting analysis...")
        output, status = self.run_command_with_output(cmd, "Project Analysis")
        progress(1.0, "Analysis complete")
        return output, status

    def run_queue(
        self, config_file: str, no_resume: bool, progress=gr.Progress()
    ) -> tuple[str, str]:
        """Run model queue training."""
        cmd = ["timeflies", "queue"]
        if config_file.strip():
            cmd.append(config_file)
        if no_resume:
            cmd.append("--no-resume")

        progress(0.1, "Starting queue training...")
        output, status = self.run_command_with_output(cmd, "Model Queue Training")
        progress(1.0, "Queue training complete")
        return output, status

    def run_verify(self, progress=gr.Progress()) -> tuple[str, str]:
        """Run system verification."""
        progress(0.1, "Verifying system...")
        output, status = self.run_command_with_output(
            ["timeflies", "verify"], "System Verification"
        )
        progress(1.0, "Verification complete")
        return output, status

    def check_batch_environment(self) -> str:
        """Check batch correction environment status."""
        if not Path(".venv_batch").exists():
            return "❌ Batch environment not found - run 'timeflies setup --dev'"

        try:
            result = subprocess.run(
                [
                    ".venv_batch/bin/python",
                    "-c",
                    "import scvi; import scib; print('OK')",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                return "✅ Batch environment ready - scVI and scib available"
            else:
                return "⚠️ Batch environment found but dependencies missing"
        except Exception as e:
            return f"❌ Error checking batch environment: {str(e)[:50]}..."

    def create_interface(self):
        """Create the Gradio interface."""
        with gr.Blocks(
            title="TimeFlies - ML for Aging Analysis",
            theme=gr.themes.Soft(),
            css="""
                .gradio-container {
                    font-family: 'Arial', sans-serif;
                }
                .main-header {
                    text-align: center;
                    color: #2c3e50;
                    margin-bottom: 20px;
                }
            """,
        ) as interface:
            # Header
            gr.Markdown(
                """
                # 🧬 TimeFlies - Machine Learning for Aging Analysis
                ### A comprehensive ML framework for Drosophila single-cell RNA sequencing data
                """,
                elem_classes=["main-header"],
            )

            # Installation Status
            with gr.Row():
                installation_status = gr.Textbox(
                    label="Installation Status",
                    value=self.check_installation_status(),
                    interactive=False,
                )
                refresh_btn = gr.Button("🔄 Refresh Status", variant="secondary")
                refresh_btn.click(
                    self.check_installation_status, outputs=installation_status
                )

            # Main tabs
            with gr.Tabs():
                # Setup Tab
                with gr.Tab("🚀 Setup & Installation"):
                    gr.Markdown("### Initial Setup and Data Preparation")

                    with gr.Row():
                        setup_btn = gr.Button("Run Setup", variant="primary", size="lg")

                    setup_output = gr.Textbox(
                        label="Setup Output", lines=15, max_lines=20, interactive=False
                    )
                    setup_status = gr.Textbox(label="Status", interactive=False)

                    setup_btn.click(
                        self.run_setup,
                        outputs=[setup_output, setup_status],
                        show_progress=True,
                    )

                # Training Tab
                with gr.Tab("🎯 Model Training"):
                    gr.Markdown("### Train Machine Learning Models")

                    with gr.Row():
                        with gr.Column():
                            include_eda = gr.Checkbox(label="Include EDA", value=True)
                            include_analysis = gr.Checkbox(
                                label="Include Analysis", value=True
                            )
                            batch_corrected = gr.Checkbox(
                                label="Use Batch Corrected Data", value=False
                            )

                    train_btn = gr.Button(
                        "Start Training", variant="primary", size="lg"
                    )

                    train_output = gr.Textbox(
                        label="Training Output",
                        lines=15,
                        max_lines=20,
                        interactive=False,
                    )
                    train_status = gr.Textbox(label="Status", interactive=False)

                    train_btn.click(
                        self.run_train,
                        inputs=[include_eda, include_analysis, batch_corrected],
                        outputs=[train_output, train_status],
                        show_progress=True,
                    )

                # Batch Correction Tab
                with gr.Tab("⚖️ Batch Correction"):
                    gr.Markdown("### Remove Technical Batch Effects with scVI")

                    with gr.Row():
                        with gr.Column():
                            project_choice = gr.Dropdown(
                                choices=["fruitfly_aging", "fruitfly_alzheimers"],
                                value="fruitfly_alzheimers",
                                label="Project",
                            )
                            tissue_choice = gr.Dropdown(
                                choices=["head", "body"], value="head", label="Tissue"
                            )

                    with gr.Row():
                        batch_env_status = gr.Textbox(
                            label="Batch Environment Status",
                            value=self.check_batch_environment(),
                            interactive=False,
                        )
                        check_env_btn = gr.Button("🔄 Check Environment")

                    batch_btn = gr.Button(
                        "Run Batch Correction", variant="primary", size="lg"
                    )

                    batch_output = gr.Textbox(
                        label="Batch Correction Output",
                        lines=15,
                        max_lines=20,
                        interactive=False,
                    )
                    batch_status = gr.Textbox(label="Status", interactive=False)

                    check_env_btn.click(
                        self.check_batch_environment, outputs=batch_env_status
                    )

                    batch_btn.click(
                        self.run_batch_correction,
                        inputs=[project_choice, tissue_choice],
                        outputs=[batch_output, batch_status],
                        show_progress=True,
                    )

                    gr.Markdown("""
                    **About Batch Correction:**
                    - Removes technical batch effects while preserving biological variation
                    - Uses scVI (single-cell Variational Inference) deep learning model
                    - Trains on train data only, applies to eval data as query (prevents data leakage)
                    - Automatically switches to batch environment (.venv_batch) with PyTorch + scVI
                    - Creates *_batch.h5ad files for downstream analysis
                    """)

                # Hyperparameter Tuning Tab
                with gr.Tab("🎛️ Hyperparameter Tuning"):
                    gr.Markdown("### Optimize Model Performance")

                    tune_btn = gr.Button(
                        "Start Hyperparameter Tuning", variant="primary", size="lg"
                    )

                    tune_output = gr.Textbox(
                        label="Tuning Output", lines=15, max_lines=20, interactive=False
                    )
                    tune_status = gr.Textbox(label="Status", interactive=False)

                    tune_btn.click(
                        self.run_hyperparameter_tuning,
                        outputs=[tune_output, tune_status],
                        show_progress=True,
                    )

                # Evaluation Tab
                with gr.Tab("📊 Model Evaluation"):
                    gr.Markdown("### Evaluate Trained Models")

                    with gr.Row():
                        with gr.Column():
                            eval_include_eda = gr.Checkbox(
                                label="Include EDA", value=True
                            )
                            eval_include_analysis = gr.Checkbox(
                                label="Include Analysis", value=True
                            )
                            eval_batch_corrected = gr.Checkbox(
                                label="Use Batch Corrected Data", value=False
                            )

                    eval_btn = gr.Button("Run Evaluation", variant="primary", size="lg")

                    eval_output = gr.Textbox(
                        label="Evaluation Output",
                        lines=15,
                        max_lines=20,
                        interactive=False,
                    )
                    eval_status = gr.Textbox(label="Status", interactive=False)

                    eval_btn.click(
                        self.run_evaluate,
                        inputs=[
                            eval_include_eda,
                            eval_include_analysis,
                            eval_batch_corrected,
                        ],
                        outputs=[eval_output, eval_status],
                        show_progress=True,
                    )

                # Analysis Tab
                with gr.Tab("🔬 Project Analysis"):
                    gr.Markdown("### Run Project-Specific Analysis")

                    with gr.Row():
                        analyze_batch_corrected = gr.Checkbox(
                            label="Use Batch Corrected Data", value=False
                        )

                    analyze_btn = gr.Button(
                        "Run Analysis", variant="primary", size="lg"
                    )

                    analyze_output = gr.Textbox(
                        label="Analysis Output",
                        lines=15,
                        max_lines=20,
                        interactive=False,
                    )
                    analyze_status = gr.Textbox(label="Status", interactive=False)

                    analyze_btn.click(
                        self.run_analyze,
                        inputs=[analyze_batch_corrected],
                        outputs=[analyze_output, analyze_status],
                        show_progress=True,
                    )

                # Queue Training Tab
                with gr.Tab("🚂 Queue Training"):
                    gr.Markdown("### Automated Multi-Model Training")

                    with gr.Row():
                        with gr.Column():
                            queue_config = gr.Textbox(
                                label="Queue Config File",
                                value="configs/model_queue.yaml",
                                placeholder="configs/model_queue.yaml",
                            )
                            queue_no_resume = gr.Checkbox(
                                label="Start Fresh (No Resume)", value=False
                            )

                    queue_btn = gr.Button(
                        "Start Queue Training", variant="primary", size="lg"
                    )

                    queue_output = gr.Textbox(
                        label="Queue Training Output",
                        lines=15,
                        max_lines=20,
                        interactive=False,
                    )
                    queue_status = gr.Textbox(label="Status", interactive=False)

                    queue_btn.click(
                        self.run_queue,
                        inputs=[queue_config, queue_no_resume],
                        outputs=[queue_output, queue_status],
                        show_progress=True,
                    )

                # System Tab
                with gr.Tab("🔧 System Management"):
                    gr.Markdown("### Update and System Operations")

                    with gr.Row():
                        update_btn = gr.Button(
                            "Update TimeFlies", variant="secondary", size="lg"
                        )
                        verify_btn = gr.Button(
                            "Verify System", variant="secondary", size="lg"
                        )

                    system_output = gr.Textbox(
                        label="System Output", lines=15, max_lines=20, interactive=False
                    )
                    system_status = gr.Textbox(label="Status", interactive=False)

                    update_btn.click(
                        self.run_update,
                        outputs=[system_output, system_status],
                        show_progress=True,
                    )

                    verify_btn.click(
                        self.run_verify,
                        outputs=[system_output, system_status],
                        show_progress=True,
                    )

                    gr.Markdown("""
                    **System Information:**
                    - Web GUI works in any browser - no system dependencies needed
                    - All operations run in your TimeFlies virtual environment
                    - Safe to close browser - operations continue running
                    - Access remotely via the provided URL (if needed)
                    """)

            # Footer
            gr.Markdown("""
            ---
            **TimeFlies v1.0** - Built by Singh Lab |
            [Documentation](https://github.com/rsinghlab/TimeFlies) |
            Report issues via GitHub
            """)

        return interface


def launch_gui(
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
    debug: bool = False,
) -> None:
    """Launch the TimeFlies web GUI."""
    try:
        gui = TimeFliesWebGUI()
        interface = gui.create_interface()

        print("🚀 Starting TimeFlies Web GUI...")
        print(f"📍 Local URL: http://{server_name}:{server_port}")

        if share:
            print("🌐 Creating public URL...")

        interface.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            debug=debug,
            show_error=True,
            quiet=False,
        )

    except Exception as e:
        print(f"❌ Failed to start GUI: {e}")
        print("💡 Make sure you're in the TimeFlies virtual environment")
        sys.exit(1)


if __name__ == "__main__":
    launch_gui()
