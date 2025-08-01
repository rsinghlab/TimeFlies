# config.py
config_dict = {
    "Device": {
        "processor": "Other",  # Options: 'Other' or 'M' (M is for Mac M1/M2/M3 processors)
    },
    "FileLocations": {
        "training_file": "fly_train.h5ad",       # Name of the training data file
        "evaluation_file": "fly_eval.h5ad",      # Name of the evaluation data file
        "original_file": "fly_original.h5ad",     # Name of the original data file
        "batch_corrected_files": {                # Batch-corrected file locations
            "train": "fly_train_batch.h5ad",
            "eval": "fly_eval_batch.h5ad",
            "original": "fly_original_batch.h5ad",
        },
    },
    "DataParameters": {
        "GeneralSettings": {
            "tissue": "head",  # Options: 'head', 'body', 'all'
            "model_type": "CNN",  # Options: 'CNN', 'MLP', 'XGBoost', 'RandomForest', 'LogisticRegression'
            "encoding_variable": "age",  # Options: 'sex_age', 'sex', 'age'
            "cell_type": "all",  # Options: 'all', 'CNS neuron', 'sensory neuron', 'epithelial cell', 'muscle cell', 'glial cell'
            "sex_type": "all",         # Options: 'all', 'male', 'female'
        },
        "Sampling": {
            "num_samples":289981,       # Number of samples (cells) for training (total = 289981)
            "num_variables":15992,    # Number of variables (genes) for training (total = 15992)
        },
        "Filtering": {
            "include_mixed_sex": False,  # Options: True, False
        },
        "BatchCorrection": {
            "enabled": True,          # Options: True, False
        },
        "TrainTestSplit": {             # Cross training and testing (ex. male train/female test)
            "method": "encoding_variable",        # Options: 'encoding_variable' (no cross testing), 'sex', tissue'.
            "train": {
                "sex": "male",           # Options: 'Male', 'Female', 'All'
                "tissue": "head",        # Options: 'Head', 'Body', 'All'
            },
            "test": {
                "sex": "female",           # Options: 'Male', 'Female', 'All'
                "tissue": "body",        # Options: 'Head', 'Body', 'All'
                "size": 0.4,             # Fraction of data for testing (if crashing due to computation, reduce this value)
            },
        },
    },
    "DataProcessing": {
        "Normalization": {
            "enabled": False,         # Options: True, False
        },
        "ModelManagement": {
            "load_model": False,      # Options: True, False
        },
        "Preprocessing": {
            "required": True,         # Options: True, False
            "save_data": True,        # Options: True, False
        },
        "ExploratoryDataAnalysis": {
            "enabled": False,         # Options: True, False
        },
    },
    "GenePreprocessing": {
        "GeneFiltering": {
            "remove_sex_genes": False,          # Options: True, False
            "remove_autosomal_genes": False,    # Options: True, False
            "only_keep_lnc_genes": False,       # Options: True, False
            "remove_lnc_genes": False,          # Options: True, False
            "remove_unaccounted_genes": False,   # Options: True, False
            "select_batch_genes": False,        # Options: True, False, all other gene preprocessing options are ignored if this is True
            "highly_variable_genes": False,     # Options: True, False, all other gene preprocessing options are ignored if this is True
        },
        "GeneBalancing": {
            "balance_genes": False,             # Options: True, False (Must set remove_sex_genes to True)
            "balance_lnc_genes": False,         # Options: True, False
        },
        "GeneShuffle": {
            "shuffle_genes": False,             # Options: True, False
            'shuffle_random_state': 42,        # Random state for shuffling
        },
    },
    "FeatureImportanceAndVisualizations": {
        "run_visualization": True,       # Options: True, False
        "run_interpreter": True,        # Options: True, False (SHAP)
        "load_SHAP": True,              # Options: True to load SHAP values, False to compute them, only works if run_interpreter is True
        "reference_size": 5000,          # Reference data size for SHAP
        "save_predictions": False,        # Options: True, False; (Model predictions csv file)
    },
    "DataSplit": {
        "validation_split": 0.1,           # Fraction of data for validation
        "test_split": 0.1,                 # Fraction of data for testing
        "random_state": 100,               # Random state for reproducibility
    },
    "Training": {
        "epochs": 15,                      # Number of epochs for training
        "batch_size": 250,                 # Batch size for training
        "early_stopping_patience": 3,      # Patience for early stopping
        "custom_loss": "categorical_crossentropy",  # Custom loss function
        "metrics": ["accuracy", "AUC"],    # Metrics to evaluate
    },
    "ModelParameters": {
        "MLP_Model": {
            "units": [128, 256, 128],          # Number of units in each layer
            "dropout_rate": 0.3,               # Dropout rate
            "learning_rate": 0.0006,           # Learning rate
            "activation_function": "relu",     # Activation function
         
        },

        "CNN_Model": { # Only one convolutional layer
            "filters": [32],               # Only one convolutional layer
            "kernel_sizes": [3],           # Corresponding kernel size
            "strides": [1],
            "paddings": ["same"],
            "pool_sizes": [2],
            "pool_strides": [2],
            
            "dense_units": [128],          # Only one fully connected layer before output
            "dropout_rate": 0.5,
            "learning_rate": 0.001,
            "activation_function": "relu"
        },

        "XGBoost_Model": {
            "learning_rate": 0.1,               # Learning rate
            "max_depth": 6,                     # Maximum depth of trees
            "min_child_weight": 1,              # Minimum sum of instance weight needed in a child
            "subsample": 1,                     # Subsample ratio of the training instance
            "colsample_bytree": 1,              # Subsample ratio of columns when constructing each tree
            "early_stopping_rounds": 10,        # Early stopping rounds
            "seed": 42,                         # Random seed
            "tree_method": "gpu_hist",          # Options: 'auto', 'exact', 'approx', 'hist', 'gpu_hist'
            "predictor": "gpu_predictor",       # Options: 'auto', 'cpu_predictor', 'gpu_predictor'
            "n_estimators": 100,                # Number of trees
            "random_state": 42,                 # Random state
        },
        "RandomForest_Model": {
            "n_estimators": 100,                # Number of trees
            "criterion": "gini",                # Options: 'gini', 'entropy'
            "max_depth": None,                  # Maximum depth of trees
            "min_samples_split": 2,             # Minimum samples to split a node
            "min_samples_leaf": 1,              # Minimum samples at a leaf node
            "max_features": "sqrt",             # Number of features to consider when looking for the best split
            "bootstrap": True,                  # Whether bootstrap samples are used when building trees
            "oob_score": True,                  # Whether to use out-of-bag samples to estimate the R² score
            "n_jobs": -1,                        # Number of jobs to run in parallel (-1 means using all processors)
            "random_state": 42,                 # Random state
        },
        "LogisticRegression_Model": {
            "penalty": "l1",                    # Options: 'l1', 'l2', 'elasticnet'
            "solver": "liblinear",              # Options: 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
            "l1_ratio": 0.5,                    # Only used if penalty is 'elasticnet'
            "max_iter": 100,                    # Maximum number of iterations
            "random_state": 42,                 # Random state
        },
    },
    "Setup": {
        "strata": "age",                        # Column used for stratification
        "tissue": "head",                       # Tissue type (e.g., 'head', 'body')
        "seed": 42,                             # Random seed for reproducibility
        "split_size": 5000,                     # Number of samples to split for evaluation
    },
}

class ConfigHandler:
    def __init__(self, config_dict):
        self._config_dict = {}
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = ConfigHandler(value)
            self._config_dict[key] = value

    def __getattr__(self, name):
        try:
            return self._config_dict[name]
        except KeyError:
            raise AttributeError(f"No such attribute: {name}")

    def __setattr__(self, name, value):
        if name == "_config_dict":
            super().__setattr__(name, value)
        else:
            if isinstance(value, dict):
                value = ConfigHandler(value)
            self._config_dict[name] = value

    def __getitem__(self, key):
        return self._config_dict[key]

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = ConfigHandler(value)
        self._config_dict[key] = value

    def __iter__(self):
        return iter(self._config_dict)

    def items(self):
        return self._config_dict.items()

    def keys(self):
        return self._config_dict.keys()

    def values(self):
        return self._config_dict.values()

    def get(self, key, default=None):
        return self._config_dict.get(key, default)

    def as_dict(self):
        result = {}
        for key, value in self._config_dict.items():
            if isinstance(value, ConfigHandler):
                value = value.as_dict()
            result[key] = value
        return result

    def __repr__(self):
        return f"{self.__class__.__name__}({self._config_dict})"

config = ConfigHandler(config_dict)