"""Comprehensive unit tests for GPU handler."""

import pytest
import sys
from unittest.mock import Mock, patch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from shared.utils.gpu_handler import GPUHandler


class TestGPUHandler:
    """Test GPU handling utilities."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock config for testing."""
        config = Mock()
        config.device = Mock()
        return config
    
    @patch('shared.utils.gpu_handler.tf')
    def test_gpu_configure_regular_processor(self, mock_tf):
        """Test GPU configuration for regular processors."""
        # Mock TensorFlow GPU detection
        mock_tf.config.list_physical_devices.return_value = ['GPU:0']
        mock_tf.config.experimental.list_physical_devices.return_value = ['GPU:0']
        mock_tf.config.experimental.set_memory_growth = Mock()
        
        # Mock config
        from types import SimpleNamespace
        config = SimpleNamespace()
        config.device = SimpleNamespace(processor='GPU')
        
        # Should not crash
        GPUHandler.configure(config)
        
        # Should call GPU setup functions
        mock_tf.config.experimental.list_physical_devices.assert_called_with('GPU')
    
    @patch('shared.utils.gpu_handler.tf')
    def test_gpu_configure_apple_processor(self, mock_tf):
        """Test GPU configuration for Apple Silicon."""
        # Mock config for Apple M processor
        from types import SimpleNamespace
        config = SimpleNamespace()
        config.device = SimpleNamespace(processor='M')
        
        # Should not crash
        GPUHandler.configure(config)
        
        # Should call Apple-specific setup
        mock_tf.config.set_visible_devices.assert_called()
    
    @patch('shared.utils.gpu_handler.tf')
    def test_gpu_configure_no_gpu(self, mock_tf):
        """Test GPU configuration when no GPU available."""
        # Mock TensorFlow with no GPU
        mock_tf.config.experimental.list_physical_devices.return_value = []
        
        from types import SimpleNamespace
        config = SimpleNamespace()
        config.device = SimpleNamespace(processor='GPU')
        
        # Should handle no GPU gracefully
        GPUHandler.configure(config)
    
    @patch('shared.utils.gpu_handler.tf')
    def test_gpu_configure_memory_growth_error(self, mock_tf):
        """Test GPU configuration when memory growth fails."""
        # Mock GPU available but memory growth fails
        mock_tf.config.experimental.list_physical_devices.return_value = ['GPU:0']
        mock_tf.config.experimental.set_memory_growth.side_effect = RuntimeError("Memory error")
        
        from types import SimpleNamespace
        config = SimpleNamespace()
        config.device = SimpleNamespace(processor='GPU')
        
        # Should handle memory growth errors gracefully
        GPUHandler.configure(config)

    @patch('shared.utils.gpu_handler.tf')
    def test_configure_gpu_available(self, mock_tf, mock_config):
        """Test GPU configuration when GPUs are available."""
        mock_config.device.processor = 'GPU'
        
        # Mock TensorFlow GPU detection
        mock_tf.config.list_physical_devices.return_value = [Mock(), Mock()]  # 2 GPUs
        mock_tf.config.experimental.list_physical_devices.return_value = [Mock(), Mock()]
        
        GPUHandler.configure(mock_config)
        
        # Should check for GPUs
        mock_tf.config.list_physical_devices.assert_called_with("GPU")
        mock_tf.config.experimental.list_physical_devices.assert_called_with("GPU")
        
        # Should set memory growth for each GPU
        assert mock_tf.config.experimental.set_memory_growth.call_count == 2
    
    @patch('shared.utils.gpu_handler.tf')
    def test_configure_no_gpu_available(self, mock_tf, mock_config):
        """Test GPU configuration when no GPUs are available."""
        mock_config.device.processor = 'GPU'
        
        # Mock no GPUs available
        mock_tf.config.list_physical_devices.return_value = []
        mock_tf.config.experimental.list_physical_devices.return_value = []
        
        GPUHandler.configure(mock_config)
        
        # Should check for GPUs but not try to configure them
        mock_tf.config.list_physical_devices.assert_called_with("GPU")
        mock_tf.config.experimental.set_memory_growth.assert_not_called()
    
    @patch('shared.utils.gpu_handler.tf')
    def test_configure_apple_silicon(self, mock_tf, mock_config):
        """Test configuration for Apple Silicon processors."""
        mock_config.device.processor = 'M'
        
        GPUHandler.configure(mock_config)
        
        # Should configure for Apple Silicon - exact number depends on implementation
        # Just verify it was called at least once
        assert mock_tf.config.set_visible_devices.call_count >= 1
    
    @patch('shared.utils.gpu_handler.tf')
    def test_configure_apple_silicon_error(self, mock_tf, mock_config):
        """Test handling of Apple Silicon configuration errors."""
        mock_config.device.processor = 'M'
        
        # Mock set_visible_devices failure
        mock_tf.config.set_visible_devices.side_effect = Exception("Device configuration error")
        
        # Should not raise exception, just print error  
        GPUHandler.configure(mock_config)
        
        # Should have attempted to configure devices
        mock_tf.config.set_visible_devices.assert_called()
    
    def test_configure_static_method(self, mock_config):
        """Test that configure is a static method."""
        # Should be callable without instance
        assert callable(GPUHandler.configure)
        
        # Should be accessible from class
        assert hasattr(GPUHandler, 'configure')
    
    @patch('shared.utils.gpu_handler.tf')
    def test_processor_type_handling(self, mock_tf, mock_config):
        """Test different processor type handling."""
        # Test various processor values - focusing on non-Apple Silicon
        processor_types = ['GPU', 'gpu', 'CPU', 'cpu', 'OTHER']
        
        for processor in processor_types:
            mock_config.device.processor = processor
            mock_tf.reset_mock()
            
            # Regular GPU path for all non-Apple Silicon processors
            mock_tf.config.experimental.list_physical_devices.return_value = []
            mock_tf.config.list_physical_devices.return_value = []
            
            GPUHandler.configure(mock_config)
            mock_tf.config.list_physical_devices.assert_called_with("GPU")
    
    def test_config_device_attribute_access(self):
        """Test configuration device attribute access patterns."""
        config = Mock()
        config.device = Mock()
        config.device.processor = 'GPU'
        
        # Should be able to access processor attribute
        processor = getattr(config.device, 'processor', 'GPU')
        assert processor == 'GPU'
    
    @patch('shared.utils.gpu_handler.tf')  
    def test_gpu_detection_logic(self, mock_tf, mock_config):
        """Test the GPU detection logic."""
        mock_config.device.processor = 'GPU'
        
        # Test when tf.config.list_physical_devices returns different values
        test_cases = [
            ([], False),  # No GPUs detected
            ([Mock()], True),  # One GPU detected  
            ([Mock(), Mock(), Mock()], True),  # Multiple GPUs detected
        ]
        
        for gpu_list, should_configure in test_cases:
            mock_tf.reset_mock()
            mock_tf.config.list_physical_devices.return_value = gpu_list
            mock_tf.config.experimental.list_physical_devices.return_value = gpu_list
            
            GPUHandler.configure(mock_config)
            
            if should_configure and len(gpu_list) > 0:
                # Should attempt memory growth for each GPU
                assert mock_tf.config.experimental.set_memory_growth.call_count == len(gpu_list)
            else:
                # Should not attempt memory growth
                mock_tf.config.experimental.set_memory_growth.assert_not_called()
    
    def test_gpu_handler_is_class(self):
        """Test that GPUHandler is properly structured as a class."""
        assert isinstance(GPUHandler, type)
        assert hasattr(GPUHandler, 'configure')
        
        # Should not be instantiable (all methods static)
        # The configure method should be static
        configure_method = getattr(GPUHandler, 'configure')
        assert callable(configure_method)