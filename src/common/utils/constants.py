"""Constants and configuration values for TimeFlies project."""

from typing import List, Dict, Any

# Model constants
DEFAULT_TARGET_SUM = 1e4
DEFAULT_N_TOP_GENES = 5000
DEFAULT_REFERENCE_SIZE = 100
DEFAULT_RANDOM_STATE = 42

# Supported model types
SUPPORTED_MODEL_TYPES = ["cnn", "mlp", "logistic", "xgboost", "random_forest"]

# Supported encoding variables
SUPPORTED_ENCODING_VARS = ["age", "sex", "tissue", "dataset"]

# File extensions
H5AD_EXTENSION = ".h5ad"
CSV_EXTENSION = ".csv"
PKL_EXTENSION = ".pkl"
JSON_EXTENSION = ".json"
H5_EXTENSION = ".h5"

# Directory names
DATA_DIR = "Data"
MODELS_DIR = "Models"
ANALYSIS_DIR = "Analysis"
PREPROCESSED_DIR = "preprocessed"
GENE_LISTS_DIR = "gene_lists"
H5AD_DIR = "h5ad"

# Batch correction
BATCH_CORRECTED_LAYER = "scvi_normalized"
UNCORRECTED_DIR = "uncorrected"
BATCH_CORRECTED_DIR = "batch_corrected"

# Gene types
LNC_PREFIX = "lnc"
AUTOSOMAL_GENES_FILE = "autosomal.csv"
SEX_GENES_FILE = "sex.csv"

# Sex types
VALID_SEX_TYPES = ["male", "female", "mix", "all"]
VALID_CELL_TYPES = ["all"]  # Will be extended based on data

# Split methods
VALID_SPLIT_METHODS = ["sex", "tissue", "random"]

# Normalization
LOG1P_APPLIED = True
STANDARD_SCALING_APPLIED = True

# Model training
EARLY_STOPPING_PATIENCE = 10
DEFAULT_VALIDATION_SPLIT = 0.2
DEFAULT_TEST_SPLIT = 0.2
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 100

# GPU configuration
GPU_MEMORY_GROWTH = True
APPLE_SILICON_PROCESSOR = "M"

# File naming patterns
MODEL_CHECKPOINT_PATTERN = "best_model.h5"
HISTORY_FILE = "history.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"
SCALER_FILE = "scaler.pkl"
HVG_FILE = "highly_variable_genes.pkl"
METRICS_FILE = "Stats.JSON"

# Visualization
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
DEFAULT_FIGURE_SIZE = (10, 8)

# Analysis subdirectories
RESULTS_SUBDIR = "Results"
EDA_SUBDIR = "EDA"
SHAP_SUBDIR = "SHAP"
UMAP_SUBDIR = "UMAP"

# Configuration validation
REQUIRED_CONFIG_SECTIONS = [
    "DataParameters",
    "GenePreprocessing",
    "DataProcessing",
    "ModelParameters",
    "Device",
]

# Error messages
ERROR_MESSAGES = {
    "config_missing_section": "Configuration missing required section: {}",
    "invalid_model_type": f"Model type must be one of {SUPPORTED_MODEL_TYPES}",
    "invalid_sex_type": f"Sex type must be one of {VALID_SEX_TYPES}",
    "invalid_split_method": f"Split method must be one of {VALID_SPLIT_METHODS}",
    "file_not_found": "Required file not found: {}",
    "invalid_gene_balance": "Cannot balance genes: insufficient genes of type {}",
    "gpu_config_failed": "Failed to configure GPU: {}",
    "data_shape_mismatch": "Data shape mismatch between train and test sets",
    "preprocessing_failed": "Data preprocessing failed: {}",
    "model_build_failed": "Model building failed: {}",
    "model_train_failed": "Model training failed: {}",
    "evaluation_failed": "Model evaluation failed: {}",
}
