# TimeFlies User Setup Guide

## 🚀 **Recommended User Workflow**

### **For New Users (Recommended)**

1. **Create Your Analysis Folder**
   ```bash
   mkdir timeflies_analysis
   cd timeflies_analysis
   ```

2. **Get TimeFlies**
   ```bash
   # Download installer
   curl -O https://raw.githubusercontent.com/your-username/TimeFlies/code-reorganization/install_timeflies.sh
   
   # Run installation
   bash install_timeflies.sh
   
   # Activate environment  
   source activate.sh
   ```

3. **Add Your Data**
   ```bash
   # The installer creates this structure:
   mkdir -p data/fruitfly_aging/head     # or fruitfly_alzheimers
   
   # Copy your H5AD files
   cp /path/to/your_data_original.h5ad data/fruitfly_aging/head/
   ```

4. **Run Analysis**
   ```bash
   # Option A: One command does everything
   timeflies setup-all && timeflies train
   
   # Option B: Step by step
   timeflies verify
   timeflies split  
   timeflies train
   timeflies evaluate
   ```

5. **View Results**
   ```bash
   # Results are in outputs/
   ls outputs/fruitfly_aging/
   
   # Launch interactive analysis
   pip install jupyter
   jupyter notebook docs/notebooks/analysis.ipynb
   ```

### **Your Final Folder Structure**
```
timeflies_analysis/               # Your working directory
├── .venv/                       # Python environment
├── activate.sh                  # Environment activation
├── configs/
│   └── default.yaml            # Configuration file
├── data/                       # Your input data
│   └── fruitfly_aging/head/
│       └── your_data_original.h5ad
├── docs/
│   └── notebooks/
│       └── analysis.ipynb     # Interactive analysis
└── outputs/                   # All results
    └── fruitfly_aging/
        ├── models/           # Trained models
        ├── results/          # SHAP analysis, plots
        └── logs/             # Training logs
```

## 🖥️ **For GUI Users**

1. **Download GUI Launcher**
   ```bash
   curl -O https://raw.githubusercontent.com/your-username/TimeFlies/code-reorganization/TimeFlies_Launcher.py
   python3 TimeFlies_Launcher.py
   ```

2. **Use GUI Tabs:**
   - **Installation**: Click "Quick Install"
   - **Run Analysis**: Select project, click "Run Complete Analysis"  
   - **View Results**: Access outputs and notebooks
   - **Help**: Built-in documentation

## 💻 **For Developers**

If you want to develop/modify TimeFlies:

```bash
# Clone the repository
git clone https://github.com/your-username/TimeFlies.git
cd TimeFlies
git checkout code-reorganization

# Install in development mode
pip install -e .

# Use timeflies commands or run_timeflies.py directly
python run_timeflies.py --help
```

## 🔄 **Updating TimeFlies**

```bash
# In your analysis folder
curl -O https://raw.githubusercontent.com/your-username/TimeFlies/code-reorganization/install_timeflies.sh
bash install_timeflies.sh  # Will update to latest version
```

## 🆘 **Troubleshooting**

- **Python 3.12+ required**: Use `python3 --version` to check
- **Permission errors**: Run with `bash` not `sh`
- **Data not found**: Ensure files are named `*_original.h5ad`
- **Memory issues**: Use smaller batch sizes in `configs/default.yaml`

## ✅ **What You Get**

- ✅ **Complete ML Pipeline**: Data prep → Training → Evaluation → Analysis
- ✅ **SHAP Interpretability**: Understand what your models learned
- ✅ **Rich Visualizations**: Automatic plots and analysis reports  
- ✅ **Jupyter Integration**: Interactive analysis notebooks
- ✅ **Professional Results**: Publication-ready outputs
- ✅ **Easy Updates**: Simple reinstallation process