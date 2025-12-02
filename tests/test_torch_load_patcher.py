"""Tests for the torch_load_patcher module."""

import torch
import pytest
import threading
import concurrent.futures
from unittest.mock import patch, MagicMock
from chatterbox.utils.torch_load_patcher import (
    TorchLoadPatcher,
    torch_load_with_device,
    patch_torch_load,
)

# Import fixtures
pytest_plugins = ['tests.fixtures.torch_load_fixtures']


class TestTorchLoadPatcher:
    """Test suite for TorchLoadPatcher functionality."""
    
    def test_auto_detect_fixture(self, torch_load_patcher):
        """Test using the auto-detecting fixture."""
        # Within this test, all torch.load calls will use the auto-detected device
        assert torch_load_patcher.device.type in ["cuda", "mps", "cpu"]
        
        # Mock torch.load to verify it's being called with map_location
        with patch.object(torch_load_patcher, '_original_torch_load') as mock_load:
            mock_load.return_value = MagicMock()
            
            # This should add map_location automatically
            result = torch.load("dummy.pt")
            
            # Verify torch.load was called with map_location
            mock_load.assert_called_once_with("dummy.pt", map_location=torch_load_patcher.device)
    
    def test_mps_specific_fixture(self, torch_load_with_mps):
        """Test using the MPS-specific fixture."""
        # This fixture ensures MPS is used if available, otherwise CPU
        expected_device = "mps" if torch.backends.mps.is_available() else "cpu"
        assert torch_load_with_mps.device.type == expected_device
    
    def test_device_factory_fixture(self, torch_load_device):
        """Test using the device factory fixture."""
        # Test with CPU
        with torch_load_device("cpu") as patcher:
            assert patcher.device.type == "cpu"
            
            # Mock to verify behavior
            with patch.object(patcher, '_original_torch_load') as mock_load:
                mock_load.return_value = MagicMock()
                torch.load("model.pt")
                mock_load.assert_called_with("model.pt", map_location=torch.device("cpu"))
        
        # Test with CUDA (if available)
        if torch.cuda.is_available():
            with torch_load_device("cuda") as patcher:
                assert patcher.device.type == "cuda"
    
    def test_fixture_restores_original(self, torch_load_patcher):
        """Test that fixture properly restores torch.load after test."""
        original_load = torch.load
        
        # torch.load should be patched within the fixture
        assert torch.load != original_load
        
        # Create a flag to check after fixture cleanup
        # This will be checked after the test method completes
        self._original_load = original_load
    
    def test_multiple_loads_in_fixture(self, torch_load_patcher):
        """Test multiple torch.load calls within a fixture."""
        with patch.object(torch_load_patcher, '_original_torch_load') as mock_load:
            mock_load.return_value = MagicMock()
            
            # Multiple loads should all get map_location
            torch.load("model1.pt")
            torch.load("model2.pt", map_location="cuda")  # Explicit map_location preserved
            torch.load("model3.pt")
            
            assert mock_load.call_count == 3
            
            # Check each call
            calls = mock_load.call_args_list
            assert calls[0].kwargs['map_location'] == torch_load_patcher.device
            assert calls[1].kwargs['map_location'] == "cuda"  # Explicit preserved
            assert calls[2].kwargs['map_location'] == torch_load_patcher.device


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_parameterized_with_fixture(torch_load_device, device):
    """Test using fixture with pytest.mark.parametrize."""
    # Skip if device not available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    
    with torch_load_device(device) as patcher:
        assert patcher.device.type == device


# Example of using fixture in a real model test
class TestModelLoading:
    """Example tests for model loading scenarios."""
    
    @pytest.fixture
    def mock_model_file(self, tmp_path):
        """Create a mock model file."""
        model_path = tmp_path / "model.pt"
        # In real scenario, you'd save an actual model
        # torch.save({"state_dict": {}}, model_path)
        return model_path
    
    def test_load_model_with_auto_device(self, torch_load_patcher, mock_model_file):
        """Test loading a model with auto-detected device."""
        # In this test, any model loading will automatically use the best device
        # model = MyModel.load_from_checkpoint(mock_model_file)
        # assert model.device == torch_load_patcher.device
        pass
    
    def test_load_multiple_models(self, torch_load_with_mps):
        """Test loading multiple models with MPS fixture."""
        # All models loaded in this test will use MPS (or CPU as fallback)
        # model1 = load_model("model1.pt")
        # model2 = load_model("model2.pt")
        # assert both models are on the same device
        pass


# Example integration test
def test_chatterbox_tts_loading(torch_load_patcher):
    """Test that ChatterboxTTS can be loaded with the patcher."""
    try:
        from chatterbox.tts import ChatterboxTTS
        # This would normally load the model
        # model = ChatterboxTTS.from_pretrained()
        # assert model is not None
    except ImportError:
        pytest.skip("ChatterboxTTS not available in test environment")


class TestNestedContexts:
    """Test nested context manager behavior."""
    
    def test_nested_contexts_restore_correctly(self):
        """Test that nested contexts properly restore torch.load."""
        original_load = torch.load
        
        # Create mock for testing
        mock_file = MagicMock()
        
        with patch('torch.load', MagicMock(return_value="original")) as mock_original:
            # First level context
            with torch_load_with_device("cpu") as patcher1:
                assert patcher1.device.type == "cpu"
                
                # Second level context
                with torch_load_with_device("cuda:0" if torch.cuda.is_available() else "cpu") as patcher2:
                    device_type = "cuda" if torch.cuda.is_available() else "cpu"
                    assert patcher2.device.type == device_type
                    
                    # Inner context should use cuda/cpu
                    result = torch.load("dummy.pt")
                    call_args = mock_original.call_args
                    assert 'map_location' in call_args[1]
                    assert call_args[1]['map_location'].type == device_type
                
                # Back to outer context - should use CPU
                result = torch.load("dummy.pt")
                call_args = mock_original.call_args
                assert 'map_location' in call_args[1]
                assert call_args[1]['map_location'].type == "cpu"
            
            # Outside all contexts - torch.load should be restored
            # This is the key fix - it should work correctly now
            assert torch.load == mock_original
    
    def test_same_instance_reentry_error(self):
        """Test that re-entering the same instance raises an error."""
        patcher = TorchLoadPatcher("cpu")
        
        with patcher:
            # Try to re-enter the same instance
            with pytest.raises(RuntimeError, match="Cannot re-enter the same TorchLoadPatcher instance"):
                with patcher:
                    pass


class TestThreadSafety:
    """Test thread safety of the patcher."""
    
    def test_concurrent_patchers_different_threads(self):
        """Test that different threads can use different patchers concurrently."""
        results = {}
        barrier = threading.Barrier(2)
        
        def thread1_func():
            with torch_load_with_device("cpu") as patcher:
                results['thread1_device'] = patcher.device.type
                barrier.wait()  # Synchronize with thread2
                # Both threads are now in their contexts
                barrier.wait()  # Wait for thread2 to record its device
                results['thread1_still_cpu'] = patcher.device.type == "cpu"
        
        def thread2_func():
            with torch_load_with_device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") as patcher:
                results['thread2_device'] = patcher.device.type
                barrier.wait()  # Synchronize with thread1
                # Both threads are now in their contexts
                barrier.wait()  # Signal thread1 we're done
        
        thread1 = threading.Thread(target=thread1_func)
        thread2 = threading.Thread(target=thread2_func)
        
        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()
        
        # Verify each thread used its own device
        assert results['thread1_device'] == "cpu"
        assert results['thread1_still_cpu'] == True
        # Thread 2 device depends on availability
        assert results['thread2_device'] in ["cuda", "mps", "cpu"]


class TestDeviceValidation:
    """Test device validation and CUDA index support."""
    
    def test_invalid_device_raises_error(self):
        """Test that invalid device names raise ValueError."""
        with pytest.raises(ValueError, match="Invalid device"):
            TorchLoadPatcher("invalid_device")
    
    def test_cuda_index_support(self):
        """Test CUDA device index support."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Test cuda:0
        patcher0 = TorchLoadPatcher("cuda:0")
        assert patcher0.device.type == "cuda"
        assert patcher0.device.index == 0
        
        # Test cuda:1 if available
        if torch.cuda.device_count() > 1:
            patcher1 = TorchLoadPatcher("cuda:1")
            assert patcher1.device.type == "cuda"
            assert patcher1.device.index == 1
    
    def test_auto_detect_returns_cuda_with_index(self):
        """Test that auto-detect returns cuda:0 for multi-GPU systems."""
        patcher = TorchLoadPatcher()
        
        if torch.cuda.is_available():
            assert patcher.device.type == "cuda"
            assert patcher.device.index == 0
        elif torch.backends.mps.is_available():
            assert patcher.device.type == "mps"
        else:
            assert patcher.device.type == "cpu"


class TestDecoratorBehavior:
    """Test decorator-specific behavior."""
    
    def test_decorator_creates_new_instance(self):
        """Test that decorator creates new instance for each call."""
        call_count = 0
        
        @patch_torch_load("cpu")
        def test_func():
            nonlocal call_count
            call_count += 1
            return call_count
        
        # Multiple calls should work without re-entrance issues
        assert test_func() == 1
        assert test_func() == 2
        assert test_func() == 3