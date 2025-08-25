#!/usr/bin/env python3
"""
TimeFlies GUI Launcher
A user-friendly graphical interface for TimeFlies installation and management.

This provides a simple GUI alternative to command-line usage.
"""

import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk


class TimeFliesLauncher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("TimeFlies - ML for Aging Analysis")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")

        # Style
        style = ttk.Style()
        style.theme_use("clam")

        self.setup_ui()

    def setup_ui(self):
        # Main title
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        title_frame.pack(fill="x", pady=(0, 20))
        title_frame.pack_propagate(False)

        title_label = tk.Label(
            title_frame,
            text="TimeFlies - Machine Learning for Aging Analysis",
            font=("Arial", 16, "bold"),
            fg="white",
            bg="#2c3e50",
        )
        title_label.pack(expand=True)

        # Notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=20, pady=10)

        # Installation tab
        install_frame = ttk.Frame(notebook)
        notebook.add(install_frame, text="Installation")
        self.setup_install_tab(install_frame)

        # Analysis tab
        analysis_frame = ttk.Frame(notebook)
        notebook.add(analysis_frame, text="Run Analysis")
        self.setup_analysis_tab(analysis_frame)

        # Hyperparameter Tuning tab
        tuning_frame = ttk.Frame(notebook)
        notebook.add(tuning_frame, text="Hyperparameter Tuning")
        self.setup_tuning_tab(tuning_frame)

        # Results tab
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="View Results")
        self.setup_results_tab(results_frame)

        # Help tab
        help_frame = ttk.Frame(notebook)
        notebook.add(help_frame, text="Help")
        self.setup_help_tab(help_frame)

    def setup_install_tab(self, parent):
        """Setup installation tab with user-friendly options."""
        # Status
        status_frame = ttk.LabelFrame(parent, text="Installation Status", padding=10)
        status_frame.pack(fill="x", padx=10, pady=10)

        self.status_label = tk.Label(
            status_frame,
            text="Ready to install TimeFlies",
            font=("Arial", 11),
            fg="#27ae60",
        )
        self.status_label.pack()

        # Installation options
        options_frame = ttk.LabelFrame(parent, text="Installation Options", padding=10)
        options_frame.pack(fill="x", padx=10, pady=10)

        ttk.Button(
            options_frame,
            text="Quick Install (Recommended)",
            command=self.quick_install,
            width=30,
        ).pack(pady=5)

        ttk.Button(
            options_frame,
            text="Choose Installation Directory",
            command=self.custom_install,
            width=30,
        ).pack(pady=5)

        ttk.Button(
            options_frame,
            text="Verify Existing Installation",
            command=self.verify_install,
            width=30,
        ).pack(pady=5)

        # Progress
        self.progress_var = tk.StringVar(value="Ready")
        progress_frame = ttk.LabelFrame(parent, text="Progress", padding=10)
        progress_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.progress_bar = ttk.Progressbar(progress_frame, mode="indeterminate")
        self.progress_bar.pack(fill="x", pady=(0, 10))

        self.log_text = scrolledtext.ScrolledText(
            progress_frame, height=10, wrap=tk.WORD, state="disabled"
        )
        self.log_text.pack(fill="both", expand=True)

    def setup_analysis_tab(self, parent):
        """Setup analysis tab for running ML workflows."""
        # Project selection
        project_frame = ttk.LabelFrame(parent, text="Project Configuration", padding=10)
        project_frame.pack(fill="x", padx=10, pady=10)

        tk.Label(project_frame, text="Select Project:").pack(anchor="w")
        self.project_var = tk.StringVar(value="fruitfly_aging")
        ttk.Combobox(
            project_frame,
            textvariable=self.project_var,
            values=["fruitfly_aging", "fruitfly_alzheimers"],
            state="readonly",
            width=20,
        ).pack(anchor="w", pady=(0, 10))

        tk.Label(project_frame, text="Data Directory:").pack(anchor="w")
        data_frame = tk.Frame(project_frame)
        data_frame.pack(fill="x")

        self.data_path = tk.StringVar(value="./data")
        tk.Entry(data_frame, textvariable=self.data_path, width=50).pack(side="left")
        ttk.Button(data_frame, text="Browse", command=self.browse_data_dir).pack(
            side="right", padx=(10, 0)
        )

        # Analysis options
        options_frame = ttk.LabelFrame(parent, text="Analysis Options", padding=10)
        options_frame.pack(fill="x", padx=10, pady=10)

        # Option checkboxes
        self.with_eda = tk.BooleanVar(value=True)
        self.with_analysis = tk.BooleanVar(value=True)
        self.interpret = tk.BooleanVar(value=False)
        self.visualize = tk.BooleanVar(value=True)
        self.save_report = tk.BooleanVar(value=True)

        options_left = tk.Frame(options_frame)
        options_left.pack(side="left", fill="x", expand=True)

        ttk.Checkbutton(
            options_left, text="Include EDA analysis", variable=self.with_eda
        ).pack(anchor="w")
        ttk.Checkbutton(
            options_left, text="Include post-analysis", variable=self.with_analysis
        ).pack(anchor="w")
        ttk.Checkbutton(
            options_left, text="Save EDA report", variable=self.save_report
        ).pack(anchor="w")

        options_right = tk.Frame(options_frame)
        options_right.pack(side="right", fill="x", expand=True)

        ttk.Checkbutton(
            options_right, text="Enable SHAP interpretation", variable=self.interpret
        ).pack(anchor="w")
        ttk.Checkbutton(
            options_right, text="Enable visualizations", variable=self.visualize
        ).pack(anchor="w")

        # Analysis actions
        actions_frame = ttk.LabelFrame(parent, text="Analysis Actions", padding=10)
        actions_frame.pack(fill="x", padx=10, pady=10)

        # Basic actions row
        button_frame1 = tk.Frame(actions_frame)
        button_frame1.pack(pady=5)

        ttk.Button(button_frame1, text="Verify Setup", command=self.run_verify).pack(
            side="left", padx=5
        )

        ttk.Button(button_frame1, text="Split Data", command=self.run_split).pack(
            side="left", padx=5
        )

        ttk.Button(button_frame1, text="EDA Only", command=self.run_eda_only).pack(
            side="left", padx=5
        )

        # Training and evaluation row
        button_frame2 = tk.Frame(actions_frame)
        button_frame2.pack(pady=5)

        ttk.Button(button_frame2, text="Train Model", command=self.run_train).pack(
            side="left", padx=5
        )

        ttk.Button(
            button_frame2, text="Evaluate Model", command=self.run_evaluate
        ).pack(side="left", padx=5)

        ttk.Button(button_frame2, text="Run Analysis", command=self.run_analyze).pack(
            side="left", padx=5
        )

        # Queue and complete workflow row
        button_frame3 = tk.Frame(actions_frame)
        button_frame3.pack(pady=5)

        ttk.Button(
            button_frame3,
            text="Run Model Queue",
            command=self.run_model_queue,
        ).pack(side="left", padx=5)

        ttk.Button(
            button_frame3,
            text="Run Complete Workflow",
            command=self.run_complete_analysis,
        ).pack(side="left", padx=5)

    def setup_tuning_tab(self, parent):
        """Setup hyperparameter tuning tab for model optimization."""
        # Configuration
        config_frame = ttk.LabelFrame(parent, text="Tuning Configuration", padding=10)
        config_frame.pack(fill="x", padx=10, pady=10)

        # Search method selection
        tk.Label(config_frame, text="Search Method:").pack(anchor="w")
        self.search_method = tk.StringVar(value="bayesian")
        method_frame = tk.Frame(config_frame)
        method_frame.pack(fill="x", pady=(0, 10))

        ttk.Radiobutton(
            method_frame, text="Grid Search", variable=self.search_method, value="grid"
        ).pack(side="left")
        ttk.Radiobutton(
            method_frame,
            text="Random Search",
            variable=self.search_method,
            value="random",
        ).pack(side="left", padx=(20, 0))
        ttk.Radiobutton(
            method_frame,
            text="Bayesian (Optuna)",
            variable=self.search_method,
            value="bayesian",
        ).pack(side="left", padx=(20, 0))

        # Number of trials
        trials_frame = tk.Frame(config_frame)
        trials_frame.pack(fill="x")
        tk.Label(trials_frame, text="Number of Trials:").pack(side="left")
        self.n_trials = tk.StringVar(value="20")
        tk.Spinbox(
            trials_frame, from_=5, to=100, textvariable=self.n_trials, width=10
        ).pack(side="left", padx=(10, 0))

        # Search optimizations
        opt_frame = ttk.LabelFrame(parent, text="Search Optimizations", padding=10)
        opt_frame.pack(fill="x", padx=10, pady=10)

        # Speed optimization checkboxes
        self.fast_search = tk.BooleanVar(value=True)
        self.use_subset = tk.BooleanVar(value=True)

        ttk.Checkbutton(
            opt_frame,
            text="Use reduced dataset for speed (1000 cells, 500 genes)",
            variable=self.use_subset,
        ).pack(anchor="w")
        ttk.Checkbutton(
            opt_frame, text="Skip EDA/analysis during trials", variable=self.fast_search
        ).pack(anchor="w")

        # Model configuration
        model_frame = ttk.LabelFrame(parent, text="Model Settings", padding=10)
        model_frame.pack(fill="x", padx=10, pady=10)

        tk.Label(model_frame, text="Model Type:").pack(anchor="w")
        self.tuning_model = tk.StringVar(value="CNN")
        ttk.Combobox(
            model_frame,
            textvariable=self.tuning_model,
            values=["CNN", "MLP", "xgboost", "random_forest", "logistic"],
            state="readonly",
            width=20,
        ).pack(anchor="w", pady=(0, 10))

        # Action buttons
        button_frame = tk.Frame(parent)
        button_frame.pack(fill="x", padx=10, pady=20)

        ttk.Button(
            button_frame,
            text="Start Hyperparameter Search",
            command=self.run_hyperparameter_tuning,
        ).pack(side="left", padx=5)

        ttk.Button(
            button_frame,
            text="View Tuning Results",
            command=self.view_tuning_results,
        ).pack(side="left", padx=5)

        ttk.Button(
            button_frame,
            text="Resume from Checkpoint",
            command=self.resume_tuning,
        ).pack(side="left", padx=5)

    def setup_results_tab(self, parent):
        """Setup results viewing tab."""
        results_frame = ttk.LabelFrame(parent, text="Results Browser", padding=10)
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Button(
            results_frame,
            text="Open Results Folder",
            command=self.open_results_folder,
        ).pack(pady=5)

        ttk.Button(
            results_frame,
            text="Launch Jupyter Notebook",
            command=self.launch_jupyter,
        ).pack(pady=5)

        ttk.Button(
            results_frame, text="View SHAP Analysis", command=self.view_shap
        ).pack(pady=5)

        ttk.Button(
            results_frame, text="Generate Report", command=self.generate_report
        ).pack(pady=5)

        # System management
        system_frame = ttk.LabelFrame(parent, text="System Management", padding=10)
        system_frame.pack(fill="x", padx=10, pady=10)

        ttk.Button(
            system_frame,
            text="Update TimeFlies",
            command=self.update_timeflies,
        ).pack(pady=5)

        ttk.Button(
            system_frame, text="Check System Status", command=self.run_verify
        ).pack(pady=5)

    def setup_help_tab(self, parent):
        """Setup help and documentation tab."""
        help_text = """
TimeFlies - Machine Learning for Aging Analysis

QUICK START:
1. Click "Quick Install" in the Installation tab
2. Add your H5AD data files to the data directory
3. Use "Run Complete Analysis" for end-to-end processing

WORKFLOW:
• Verify: Check that data and environment are ready
• Split: Create training and evaluation datasets
• Train: Build machine learning models
• Evaluate: Generate SHAP analysis and visualizations
• Model Queue: Run multiple models sequentially with automated comparison

DATA FORMAT:
• Place *.h5ad files in data/project_name/tissue_type/
• Files should be named *_original.h5ad
• Supports both fruitfly_aging and fruitfly_alzheimers projects

SUPPORT:
• All analysis results are saved in the outputs/ directory
• Use Jupyter notebooks for interactive analysis
• Contact your lab administrator for data access

TROUBLESHOOTING:
• Ensure Python 3.12+ is installed
• Check that data files are in the correct format
• Use "Verify Setup" to diagnose issues
        """

        help_display = scrolledtext.ScrolledText(
            parent, wrap=tk.WORD, font=("Consolas", 10), state="normal"
        )
        help_display.pack(fill="both", expand=True, padx=10, pady=10)
        help_display.insert("1.0", help_text)
        help_display.configure(state="disabled")

    def log_message(self, message):
        """Add message to log display."""
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        self.root.update()

    def run_command(self, cmd, success_msg="Command completed successfully"):
        """Run command in separate thread with progress indication."""

        def run():
            try:
                self.progress_bar.start()
                self.log_message(f"Running: {' '.join(cmd)}")

                result = subprocess.run(
                    cmd, capture_output=True, text=True, cwd=Path.cwd()
                )

                if result.returncode == 0:
                    self.log_message(success_msg)
                    if result.stdout:
                        self.log_message(f"Output: {result.stdout}")
                else:
                    self.log_message(f"Error: {result.stderr}")

            except Exception as e:
                self.log_message(f"Exception: {e}")
            finally:
                self.progress_bar.stop()

        threading.Thread(target=run, daemon=True).start()

    def quick_install(self):
        """Run quick installation."""
        self.log_message("Starting TimeFlies installation...")
        # Download and run installer
        install_cmd = [
            "bash",
            "-c",
            "curl -O https://raw.githubusercontent.com/rsinghlab/TimeFlies/main/install_timeflies.sh && bash install_timeflies.sh",
        ]
        self.run_command(install_cmd, "TimeFlies installed successfully!")

    def custom_install(self):
        """Allow user to choose installation directory."""
        directory = filedialog.askdirectory(title="Choose Installation Directory")
        if directory:
            os.chdir(directory)
            self.quick_install()

    def verify_install(self):
        """Verify existing installation."""
        self.run_command(
            ["python", "run_timeflies.py", "verify"], "Installation verified!"
        )

    def browse_data_dir(self):
        """Browse for data directory."""
        directory = filedialog.askdirectory(title="Select Data Directory")
        if directory:
            self.data_path.set(directory)

    def build_command_flags(self):
        """Build command flags based on checkbox selections."""
        flags = []
        if self.with_eda.get():
            flags.append("--with-eda")
        if self.with_analysis.get():
            flags.append("--with-analysis")
        if self.interpret.get():
            flags.append("--interpret")
        if self.visualize.get():
            flags.append("--visualize")
        return flags

    def run_verify(self):
        self.run_command(
            ["python", "run_timeflies.py", "verify"], "Setup verification complete!"
        )

    def run_split(self):
        self.run_command(
            ["python", "run_timeflies.py", "split"], "Data splitting complete!"
        )

    def run_eda_only(self):
        cmd = ["python", "run_timeflies.py", "eda"]
        if self.save_report.get():
            cmd.append("--save-report")
        self.run_command(cmd, "EDA analysis complete!")

    def run_train(self):
        cmd = ["python", "run_timeflies.py", "train"] + self.build_command_flags()
        self.run_command(cmd, "Model training complete!")

    def run_evaluate(self):
        cmd = ["python", "run_timeflies.py", "evaluate"] + self.build_command_flags()
        self.run_command(cmd, "Model evaluation complete!")

    def run_analyze(self):
        cmd = ["python", "run_timeflies.py", "analyze"] + self.build_command_flags()
        self.run_command(cmd, "Analysis complete!")

    def run_model_queue(self):
        """Run automated multi-model training queue."""
        self.run_command(
            ["python", "run_timeflies.py", "queue"], "Model queue training complete!"
        )

    def run_complete_analysis(self):
        """Run complete analysis workflow with selected options."""
        self.log_message("Starting complete analysis workflow...")

        # Build commands with user-selected options
        train_cmd = ["python", "run_timeflies.py", "train"] + self.build_command_flags()
        evaluate_cmd = [
            "python",
            "run_timeflies.py",
            "evaluate",
        ] + self.build_command_flags()

        commands = [
            (["python", "run_timeflies.py", "verify"], "Setup verified"),
            (["python", "run_timeflies.py", "split"], "Data split complete"),
            (train_cmd, "Training complete"),
            (evaluate_cmd, "Evaluation complete"),
        ]

        def run_sequence():
            self.progress_bar.start()
            for cmd, msg in commands:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        self.log_message(f"[OK] {msg}")
                    else:
                        self.log_message(f"[ERROR] Failed: {msg} - {result.stderr}")
                        break
                except Exception as e:
                    self.log_message(f"[ERROR] Error: {e}")
                    break
            else:
                self.log_message("SUCCESS: Complete analysis finished successfully!")
            self.progress_bar.stop()

        threading.Thread(target=run_sequence, daemon=True).start()

    def open_results_folder(self):
        """Open results folder in file manager."""
        try:
            if sys.platform == "darwin":  # macOS
                subprocess.run(["open", "outputs"])
            elif sys.platform == "win32":  # Windows
                subprocess.run(["explorer", "outputs"])
            else:  # Linux
                subprocess.run(["xdg-open", "outputs"])
        except Exception:
            messagebox.showinfo("Info", "Please manually open the 'outputs' folder")

    def launch_jupyter(self):
        """Launch Jupyter notebook."""
        self.run_command(
            ["jupyter", "notebook", "docs/notebooks/analysis.ipynb"],
            "Jupyter notebook launched!",
        )

    def view_shap(self):
        """View SHAP analysis results."""
        self.log_message("Opening SHAP analysis results...")
        self.open_results_folder()

    def generate_report(self):
        """Generate analysis report."""
        self.run_command(["timeflies", "analyze"], "Analysis report generated!")

    def run_hyperparameter_tuning(self):
        """Run hyperparameter tuning with selected options."""
        self.log_message("Starting hyperparameter tuning...")

        # Create temporary config with tuning settings
        import tempfile

        import yaml

        # Load current default config
        try:
            with open("configs/default.yaml") as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            messagebox.showerror(
                "Error", "Default configuration file not found. Please run setup first."
            )
            return

        # Update config with GUI settings
        config["hyperparameter_tuning"] = {
            "enabled": True,
            "method": self.search_method.get(),
            "n_trials": int(self.n_trials.get()),
        }

        # Update model type
        config["data"]["model"] = self.tuning_model.get()

        # Apply optimizations if selected
        if self.use_subset.get() or self.fast_search.get():
            config["hyperparameter_tuning"]["search_optimizations"] = {}

            if self.use_subset.get():
                config["hyperparameter_tuning"]["search_optimizations"]["data"] = {
                    "sampling": {"samples": 1000, "variables": 500}
                }

            if self.fast_search.get():
                config["hyperparameter_tuning"]["search_optimizations"].update(
                    {
                        "with_eda": False,
                        "with_analysis": False,
                        "interpret": False,
                        "visualize": False,
                    }
                )

        # Save temporary config and run tuning
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_config = f.name

        try:
            self.run_command(
                ["python", "run_timeflies.py", "tune", temp_config],
                "Hyperparameter tuning completed! Check results tab.",
            )
        finally:
            # Clean up temporary file
            import os

            try:
                os.unlink(temp_config)
            except OSError:
                pass

    def view_tuning_results(self):
        """View hyperparameter tuning results."""
        self.log_message("Opening hyperparameter tuning results...")
        try:
            project = (
                self.project_var.get()
                if hasattr(self, "project_var")
                else "fruitfly_aging"
            )
            tuning_dir = f"outputs/{project}/hyperparameter_tuning"

            if os.path.exists(tuning_dir):
                if sys.platform == "win32":
                    os.startfile(tuning_dir)
                elif sys.platform == "darwin":
                    subprocess.run(["open", tuning_dir])
                else:
                    subprocess.run(["xdg-open", tuning_dir])
            else:
                messagebox.showinfo(
                    "Info", "No hyperparameter tuning results found. Run tuning first."
                )
        except Exception as e:
            messagebox.showerror("Error", f"Could not open results: {e}")

    def resume_tuning(self):
        """Resume hyperparameter tuning from checkpoint."""
        self.log_message("Resuming hyperparameter tuning from checkpoint...")
        self.run_command(
            ["python", "run_timeflies.py", "tune"],  # Default resume behavior
            "Hyperparameter tuning resumed and completed!",
        )

    def update_timeflies(self):
        """Update TimeFlies to the latest version from GitHub."""
        self.log_message("Checking for TimeFlies updates...")

        # Confirm update with user
        import tkinter.messagebox as msgbox

        if msgbox.askyesno(
            "Update TimeFlies",
            "This will update TimeFlies to the latest version from GitHub.\n"
            "Your data and configurations will be preserved.\n\n"
            "Continue with update?",
        ):
            self.run_command(
                ["python", "run_timeflies.py", "update"],
                "TimeFlies updated successfully! Please restart the application.",
                timeout=300000,  # 5 minutes timeout for git operations
            )
        else:
            self.log_message("Update cancelled by user.")

    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


def main():
    """Main entry point."""
    app = TimeFliesLauncher()
    app.run()


if __name__ == "__main__":
    main()
