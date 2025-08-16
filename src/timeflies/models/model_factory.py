"""Model factory for creating different types of models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import numpy as np

from ..utils.logging_config import get_logger
from ..utils.exceptions import ModelError
from ..utils.constants import SUPPORTED_MODEL_TYPES

logger = get_logger(__name__)


class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, config: Any):
        """Initialize base model with configuration."""
        self.config = config
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def build(self, input_shape: Tuple[int, ...], num_classes: int) -> None:
        """Build the model architecture."""
        pass
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Any:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save the model."""
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load the model."""
        pass


class CNNModel(BaseModel):
    """Convolutional Neural Network model."""
    
    def build(self, input_shape: Tuple[int, ...], num_classes: int) -> None:
        """Build CNN architecture."""
        try:
            model_config = self.config.ModelParameters.get('cnn', {})
            
            filters = model_config.get('filters', [64, 128, 256])
            kernel_sizes = model_config.get('kernel_sizes', [3, 3, 3])
            activation = model_config.get('activation', 'relu')
            dropout_rate = model_config.get('dropout_rate', 0.2)
            
            self.model = Sequential()
            
            # First conv layer
            self.model.add(Conv1D(
                filters=filters[0],
                kernel_size=kernel_sizes[0],
                activation=activation,
                input_shape=input_shape[1:]  # Remove batch dimension
            ))
            
            # Additional conv layers
            for i in range(1, len(filters)):
                self.model.add(Conv1D(
                    filters=filters[i],
                    kernel_size=kernel_sizes[i] if i < len(kernel_sizes) else kernel_sizes[-1],
                    activation=activation
                ))
            
            # Global pooling and dense layers
            self.model.add(GlobalMaxPooling1D())
            self.model.add(Dropout(dropout_rate))
            self.model.add(Dense(128, activation=activation))
            self.model.add(Dropout(dropout_rate))
            self.model.add(Dense(num_classes, activation='softmax'))
            
            # Compile model
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info(f"CNN model built with {self.model.count_params()} parameters")
            
        except Exception as e:
            raise ModelError(f"Failed to build CNN model: {e}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Any:
        """Train the CNN model."""
        try:
            training_config = self.config.ModelParameters.get('training', {})
            
            epochs = training_config.get('epochs', 100)
            batch_size = training_config.get('batch_size', 32)
            validation_split = training_config.get('validation_split', 0.2)
            
            # Prepare validation data
            validation_data = None
            if X_val is not None and y_val is not None:
                validation_data = (X_val, y_val)
                validation_split = None
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                validation_data=validation_data,
                verbose=1
            )
            
            self.is_trained = True
            logger.info("CNN model training completed")
            return history
            
        except Exception as e:
            raise ModelError(f"Failed to train CNN model: {e}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with CNN model."""
        if not self.is_trained:
            raise ModelError("Model must be trained before making predictions")
        return np.argmax(self.model.predict(X), axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities from CNN model."""
        if not self.is_trained:
            raise ModelError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def save(self, filepath: str) -> None:
        """Save CNN model."""
        self.model.save(filepath)
        logger.info(f"CNN model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load CNN model."""
        self.model = tf.keras.models.load_model(filepath)
        self.is_trained = True
        logger.info(f"CNN model loaded from {filepath}")


class MLPModel(BaseModel):
    """Multi-Layer Perceptron model."""
    
    def build(self, input_shape: Tuple[int, ...], num_classes: int) -> None:
        """Build MLP architecture."""
        try:
            model_config = self.config.ModelParameters.get('mlp', {})
            
            hidden_layers = model_config.get('hidden_layers', [512, 256, 128])
            activation = model_config.get('activation', 'relu')
            dropout_rate = model_config.get('dropout_rate', 0.2)
            
            self.model = Sequential()
            
            # Input layer
            self.model.add(Dense(
                hidden_layers[0],
                activation=activation,
                input_shape=(input_shape[-1],)  # Flatten input
            ))
            self.model.add(Dropout(dropout_rate))
            
            # Hidden layers
            for units in hidden_layers[1:]:
                self.model.add(Dense(units, activation=activation))
                self.model.add(Dropout(dropout_rate))
            
            # Output layer
            self.model.add(Dense(num_classes, activation='softmax'))
            
            # Compile model
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info(f"MLP model built with {self.model.count_params()} parameters")
            
        except Exception as e:
            raise ModelError(f"Failed to build MLP model: {e}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Any:
        """Train the MLP model."""
        # Similar to CNN training but flatten input if needed
        if len(X_train.shape) > 2:
            X_train = X_train.reshape(X_train.shape[0], -1)
        if X_val is not None and len(X_val.shape) > 2:
            X_val = X_val.reshape(X_val.shape[0], -1)
            
        return super().train(X_train, y_train, X_val, y_val)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with MLP model."""
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        return super().predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities from MLP model."""
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        return super().predict_proba(X)
    
    def save(self, filepath: str) -> None:
        """Save MLP model."""
        self.model.save(filepath)
        logger.info(f"MLP model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load MLP model."""
        self.model = tf.keras.models.load_model(filepath)
        self.is_trained = True
        logger.info(f"MLP model loaded from {filepath}")


class LogisticRegressionModel(BaseModel):
    """Logistic Regression model using scikit-learn."""
    
    def build(self, input_shape: Tuple[int, ...], num_classes: int) -> None:
        """Build logistic regression model."""
        try:
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=self.config.get('general', {}).get('random_state', 42)
            )
            logger.info("Logistic Regression model initialized")
        except Exception as e:
            raise ModelError(f"Failed to build Logistic Regression model: {e}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """Train logistic regression model."""
        try:
            # Flatten input and convert one-hot to labels
            if len(X_train.shape) > 2:
                X_train = X_train.reshape(X_train.shape[0], -1)
            if len(y_train.shape) > 1:
                y_train = np.argmax(y_train, axis=1)
            
            self.model.fit(X_train, y_train)
            self.is_trained = True
            logger.info("Logistic Regression training completed")
        except Exception as e:
            raise ModelError(f"Failed to train Logistic Regression model: {e}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with logistic regression."""
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities from logistic regression."""
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        return self.model.predict_proba(X)
    
    def save(self, filepath: str) -> None:
        """Save logistic regression model."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Logistic Regression model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load logistic regression model."""
        import pickle
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
        logger.info(f"Logistic Regression model loaded from {filepath}")


class ModelFactory:
    """Factory class for creating different types of models."""
    
    _model_classes = {
        'cnn': CNNModel,
        'mlp': MLPModel,
        'logistic': LogisticRegressionModel,
        # Add more models as needed
    }
    
    @classmethod
    def create_model(cls, model_type: str, config: Any) -> BaseModel:
        """
        Create a model of the specified type.
        
        Args:
            model_type: Type of model to create
            config: Configuration object
            
        Returns:
            Model instance
            
        Raises:
            ModelError: If model type is not supported
        """
        model_type = model_type.lower()
        
        if model_type not in cls._model_classes:
            raise ModelError(f"Unsupported model type: {model_type}. "
                           f"Supported types: {list(cls._model_classes.keys())}")
        
        model_class = cls._model_classes[model_type]
        logger.info(f"Creating {model_type.upper()} model")
        return model_class(config)
    
    @classmethod
    def register_model(cls, name: str, model_class: type) -> None:
        """Register a new model class."""
        cls._model_classes[name.lower()] = model_class
        logger.info(f"Registered new model type: {name}")
    
    @classmethod
    def list_models(cls) -> list:
        """List available model types."""
        return list(cls._model_classes.keys())