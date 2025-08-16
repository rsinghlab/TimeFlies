# 📁 TimeFlies Folder Structure

## 🎯 **Two-Level Organization**

### **Level 1: Experiment Type** (`tissue-model-encoding`)
### **Level 2: Configuration** (`method-cells-sexes`)

## 📂 **Project Structure**

```
TimeFlies/
├── data/                           # All input and processed data
│   ├── raw/                        # Raw input data
│   │   ├── h5ad/
│   │   │   ├── head/
│   │   │   │   ├── uncorrected/
│   │   │   │   │   ├── fly_train.h5ad
│   │   │   │   │   ├── fly_eval.h5ad  
│   │   │   │   │   └── fly_original.h5ad
│   │   │   │   └── batch_corrected/
│   │   │   │       ├── fly_train_batch.h5ad
│   │   │   │       └── fly_eval_batch.h5ad
│   │   │   └── body/ (similar structure)
│   │   └── gene_lists/
│   │       ├── autosomal.csv
│   │       └── sex.csv
│   └── processed/                  # Preprocessed data
│       ├── batch_corrected/
│       │   ├── head-cnn-age/
│       │   │   ├── hvg-all_cells-all_sexes/
│       │   │   ├── full-muscle_cell-male/
│       │   │   └── lnc_only-neuron-female/
│       │   └── body-mlp-tissue/
│       │       └── hvg-all_cells-all_sexes/
│       └── uncorrected/
│           ├── head-cnn-age/
│           │   ├── no_sex-all_cells-all_sexes/
│           │   └── hvg-epithelial_cell-male/
│           └── body-mlp-age/
│               └── full-all_cells-all_sexes/
│
├── outputs/                        # All outputs
│   ├── models/                     # Trained models 
│   │   ├── batch_corrected/
│   │   │   └── head-cnn-age/
│   │   │       ├── hvg-all_cells-all_sexes/
│   │   │       │   ├── model.h5
│   │   │       │   ├── config.yaml
│   │   │       │   ├── metrics.json
│   │   │       │   └── history.pkl
│   │   │       └── full-muscle_cell-male/
│   │   └── uncorrected/
│   │       └── head-cnn-age/
│   │           └── no_sex-all_cells-all_sexes/
│   ├── results/                    # Analysis results and plots
│   │   ├── batch_corrected/
│   │   │   └── head-cnn-age/
│   │   │       └── hvg-all_cells-all_sexes/
│   │   │           ├── plots/
│   │   │           │   ├── confusion_matrix.png
│   │   │           │   ├── roc_curve.png
│   │   │           │   └── training_metrics.png
│   │   │           └── Stats.JSON
│   │   └── uncorrected/
│   └── logs/                       # Log files
│       ├── training_2024_08_16.log
│       └── pipeline_debug.log
│
├── src/timeflies/                  # New refactored code
└── run_timeflies.py               # Simple entry point
```

## 🏷️ **Naming Convention**

### **Level 1: Experiment Type**
Format: `{tissue}-{model}-{encoding_variable}`
- Examples: 
  - `head-cnn-age`
  - `body-mlp-tissue` 
  - `head-cnn-sex`

### **Level 2: Configuration Details**  
Format: `{gene_method}-{cell_type}-{sex_type}`
- Examples:
  - `hvg-all_cells-all_sexes` (most common)
  - `full-muscle_cell-male` (specific cells and sex)
  - `no_sex-epithelial_cell-female` (no sex genes, specific cells)
  - `lnc_only-all_cells-train_male-test_female` (special split)

### **Gene Methods:**
- `hvg` - Highly variable genes
- `full` - All genes  
- `no_sex` - Remove sex genes
- `no_autosomal` - Remove autosomal genes
- `lnc_only` - Only lncRNA genes
- `balanced` - Balanced autosomal/sex genes
- `batch_genes` - Batch-corrected gene selection

### **Cell Types:**
- `all_cells` (default)
- `muscle_cell` (spaces → underscores)
- `epithelial_cell`
- `cns_neuron`

### **Sex Types:** 
- `all_sexes` (default)
- `male`, `female`
- `train_male-test_female` (special splits)

## 📈 **Benefits**

### **Before (7+ levels deep):**
```
Models/uncorrected/head/CNN/age/full_data/all_cells/all_sexes/model.h5
```

### **After (5 levels max):**
```
outputs/models/uncorrected/head-cnn-age/full-all_cells-all_sexes/model.h5
```

### **Key Improvements:**
- ✅ **50% shorter paths** - Much easier to navigate
- ✅ **Logical grouping** - Related experiments together  
- ✅ **Clear naming** - Descriptive but concise
- ✅ **Easy cleanup** - Delete entire experiment folders
- ✅ **Better organization** - Input/processed/output separation
- ✅ **Consistent format** - Predictable structure

## 🔄 **Migration Script**

You can migrate your existing data structure with:

```bash
python scripts/migrate_folders.py
```

This will:
1. Create new `data/` and `outputs/` directories  
2. Move existing files to new locations
3. Update any hardcoded paths
4. Keep originals as backup until confirmed working

## 🎯 **Examples**

```python
# Configuration automatically generates paths:
config = get_config("configs/head_cnn_age.yaml")
path_manager = PathManager(config)

# Automatically creates: 
# outputs/models/batch_corrected/head-cnn-age/hvg-all_cells-all_sexes/
model_dir = path_manager.construct_model_directory()

# outputs/results/batch_corrected/head-cnn-age/hvg-all_cells-all_sexes/plots/
results_dir = path_manager.get_visualization_directory("plots")

# data/processed/batch_corrected/head-cnn-age/hvg-all_cells-all_sexes/
processed_dir = path_manager.get_processed_data_dir()
```

Much cleaner and easier to work with! 🎉