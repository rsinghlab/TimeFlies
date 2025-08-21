"""Model classes for fruitfly_alzheimers project."""

# For now, import from shared models - can be specialized later
from shared.models.model import *

class ModelBuilder:
    """Model builder for Alzheimers project - delegates to shared implementation."""
    
    def __init__(self, config):
        from shared.models.model_factory import ModelFactory
        self.factory = ModelFactory(config)
    
    def build_model(self, *args, **kwargs):
        return self.factory.create_model(*args, **kwargs)

class ModelLoader:
    """Model loader for Alzheimers project - delegates to shared implementation."""
    
    def __init__(self, config):
        self.config = config
    
    def load_model(self, *args, **kwargs):
        # Implement as needed based on shared models
        pass