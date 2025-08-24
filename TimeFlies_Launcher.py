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
            text="🧬 TimeFlies - Machine Learning for Aging Analysis",
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
        notebook.add(install_frame, text="📥 Installation")
        self.setup_install_tab(install_frame)

        # Analysis tab
        analysis_frame = ttk.Frame(notebook)
        notebook.add(analysis_frame, text="🔬 Run Analysis")
        self.setup_analysis_tab(analysis_frame)

        # Results tab
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="📊 View Results")
        self.setup_results_tab(results_frame)

        # Help tab
        help_frame = ttk.Frame(notebook)
        notebook.add(help_frame, text="❓ Help")
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
            text="🚀 Quick Install (Recommended)",
            command=self.quick_install,
            width=30,
        ).pack(pady=5)

        ttk.Button(
            options_frame,
            text="📁 Choose Installation Directory",
            command=self.custom_install,
            width=30,
        ).pack(pady=5)

        ttk.Button(
            options_frame,
            text="✅ Verify Existing Installation",
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

        # Analysis actions
        actions_frame = ttk.LabelFrame(parent, text="Analysis Actions", padding=10)
        actions_frame.pack(fill="x", padx=10, pady=10)

        button_frame = tk.Frame(actions_frame)
        button_frame.pack()

        ttk.Button(button_frame, text="🔍 Verify Setup", command=self.run_verify).pack(
            side="left", padx=5
        )

        ttk.Button(button_frame, text="✂️ Split Data", command=self.run_split).pack(
            side="left", padx=5
        )

        ttk.Button(button_frame, text="🧠 Train Model", command=self.run_train).pack(
            side="left", padx=5
        )

        ttk.Button(button_frame, text="📊 Evaluate", command=self.run_evaluate).pack(
            side="left", padx=5
        )

        # One-click analysis
        ttk.Button(
            actions_frame,
            text="🚀 Run Complete Analysis (One-Click)",
            command=self.run_complete_analysis,
            width=40,
        ).pack(pady=(10, 0))

    def setup_results_tab(self, parent):
        """Setup results viewing tab."""
        results_frame = ttk.LabelFrame(parent, text="Results Browser", padding=10)
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Button(
            results_frame,
            text="📁 Open Results Folder",
            command=self.open_results_folder,
        ).pack(pady=5)

        ttk.Button(
            results_frame,
            text="📓 Launch Jupyter Notebook",
            command=self.launch_jupyter,
        ).pack(pady=5)

        ttk.Button(
            results_frame, text="📈 View SHAP Analysis", command=self.view_shap
        ).pack(pady=5)

        ttk.Button(
            results_frame, text="📋 Generate Report", command=self.generate_report
        ).pack(pady=5)

    def setup_help_tab(self, parent):
        """Setup help and documentation tab."""
        help_text = """
🧬 TimeFlies - Machine Learning for Aging Analysis

QUICK START:
1. Click "Quick Install" in the Installation tab
2. Add your H5AD data files to the data directory
3. Use "Run Complete Analysis" for end-to-end processing

WORKFLOW:
• Verify: Check that data and environment are ready
• Split: Create training and evaluation datasets
• Train: Build machine learning models
• Evaluate: Generate SHAP analysis and visualizations

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
        self.run_command(["timeflies", "verify"], "Installation verified!")

    def browse_data_dir(self):
        """Browse for data directory."""
        directory = filedialog.askdirectory(title="Select Data Directory")
        if directory:
            self.data_path.set(directory)

    def run_verify(self):
        self.run_command(["timeflies", "verify"], "Setup verification complete!")

    def run_split(self):
        self.run_command(["timeflies", "split"], "Data splitting complete!")

    def run_train(self):
        self.run_command(["timeflies", "train"], "Model training complete!")

    def run_evaluate(self):
        self.run_command(["timeflies", "evaluate"], "Model evaluation complete!")

    def run_complete_analysis(self):
        """Run complete analysis workflow."""
        self.log_message("Starting complete analysis workflow...")
        commands = [
            (["timeflies", "verify"], "Setup verified"),
            (["timeflies", "split"], "Data split complete"),
            (["timeflies", "train"], "Training complete"),
            (["timeflies", "evaluate"], "Evaluation complete"),
        ]

        def run_sequence():
            self.progress_bar.start()
            for cmd, msg in commands:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        self.log_message(f"✅ {msg}")
                    else:
                        self.log_message(f"❌ Failed: {msg} - {result.stderr}")
                        break
                except Exception as e:
                    self.log_message(f"❌ Error: {e}")
                    break
            else:
                self.log_message("🎉 Complete analysis finished successfully!")
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

    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


def main():
    """Main entry point."""
    app = TimeFliesLauncher()
    app.run()


if __name__ == "__main__":
    main()
