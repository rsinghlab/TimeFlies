"""Utility functions for creating proper test fixtures."""

from types import SimpleNamespace


def create_test_config():
    """Create a realistic test configuration object instead of Mock objects."""
    config = SimpleNamespace()
    
    # Data configuration
    config.data = SimpleNamespace()
    config.data.tissue = 'head'
    config.data.model_type = 'CNN'
    config.data.encoding_variable = 'age'
    config.data.cell_type = 'all'
    config.data.sex_type = 'all'
    
    # Filtering configuration
    config.data.filtering = SimpleNamespace()
    config.data.filtering.include_mixed_sex = False
    
    # Sampling configuration
    config.data.sampling = SimpleNamespace()
    config.data.sampling.num_samples = None
    config.data.sampling.num_variables = None
    
    # Train-test split configuration
    config.data.train_test_split = SimpleNamespace()
    config.data.train_test_split.method = 'random'
    config.data.train_test_split.test_split = 0.2
    config.data.train_test_split.random_state = 42
    config.data.train_test_split.train = SimpleNamespace()
    config.data.train_test_split.train.sex = 'male'
    config.data.train_test_split.train.tissue = 'head'
    config.data.train_test_split.test = SimpleNamespace()
    config.data.train_test_split.test.sex = 'female'
    config.data.train_test_split.test.tissue = 'body'
    config.data.train_test_split.test.size = 0.3
    
    # Batch correction configuration
    config.data.batch_correction = SimpleNamespace()
    config.data.batch_correction.enabled = False
    
    # Gene preprocessing configuration
    config.gene_preprocessing = SimpleNamespace()
    config.gene_preprocessing.gene_shuffle = SimpleNamespace()
    config.gene_preprocessing.gene_shuffle.shuffle_genes = False
    config.gene_preprocessing.gene_shuffle.shuffle_random_state = 42
    
    config.gene_preprocessing.gene_filtering = SimpleNamespace()
    config.gene_preprocessing.gene_filtering.remove_sex_genes = False
    config.gene_preprocessing.gene_filtering.highly_variable_genes = False
    config.gene_preprocessing.gene_filtering.select_batch_genes = False
    
    config.gene_preprocessing.gene_balancing = SimpleNamespace()
    config.gene_preprocessing.gene_balancing.balance_genes = False
    
    # Data processing configuration  
    config.data_processing = SimpleNamespace()
    config.data_processing.preprocessing = SimpleNamespace()
    config.data_processing.preprocessing.required = True
    config.data_processing.preprocessing.save_data = False
    
    config.data_processing.normalization = SimpleNamespace()
    config.data_processing.normalization.enabled = False
    
    # Model configuration
    config.model = SimpleNamespace()
    config.model.training = SimpleNamespace()
    config.model.training.epochs = 2
    config.model.training.batch_size = 16
    config.model.training.validation_split = 0.2
    config.model.training.early_stopping_patience = 1
    config.model.training.learning_rate = 0.001
    
    # Feature importance configuration
    config.feature_importance = SimpleNamespace()
    config.feature_importance.run_interpreter = False
    config.feature_importance.run_visualization = False
    
    # File locations
    config.file_locations = SimpleNamespace()
    config.file_locations.training_file = "fly_train.h5ad"
    config.file_locations.evaluation_file = "fly_eval.h5ad"
    config.file_locations.original_file = "fly_original.h5ad"
    
    return config