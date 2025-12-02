"""Pytest fixtures for torch_load_patcher."""

import pytest
import torch
from typing import Union
from chatterbox.utils.torch_load_patcher import TorchLoadPatcher


@pytest.fixture
def torch_load_patcher():
    """Pytest fixture that patches torch.load for the duration of a test.
    
    This fixture automatically detects the best available device (CUDA > MPS > CPU)
    and ensures all torch.load calls within the test use that device.
    
    Example:
        def test_model_loading(torch_load_patcher):
            model = MyModel.from_pretrained()  # Will use auto-detected device
            assert model is not None
            assert torch_load_patcher.device.type in ["cuda", "mps", "cpu"]
    """
    patcher = TorchLoadPatcher()
    with patcher:
        yield patcher


@pytest.fixture
def torch_load_with_mps():
    """Pytest fixture that patches torch.load to use MPS device.
    
    Falls back to CPU if MPS is not available.
    
    Example:
        def test_mac_specific(torch_load_with_mps):
            model = MyModel.from_pretrained()
            assert torch_load_with_mps.device.type in ["mps", "cpu"]
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    patcher = TorchLoadPatcher(device)
    with patcher:
        yield patcher


@pytest.fixture
def torch_load_with_cuda():
    """Pytest fixture that patches torch.load to use CUDA device.
    
    Falls back to CPU if CUDA is not available. Uses cuda:0 by default.
    
    Example:
        def test_gpu_specific(torch_load_with_cuda):
            if torch.cuda.is_available():
                model = MyModel.from_pretrained()
                assert torch_load_with_cuda.device.type == "cuda"
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    patcher = TorchLoadPatcher(device)
    with patcher:
        yield patcher


@pytest.fixture
def torch_load_device():
    """Pytest fixture factory for creating device-specific patchers.
    
    This fixture returns a factory function that creates patchers for
    specific devices on demand.
    
    Example:
        def test_multi_device(torch_load_device):
            # Test CPU loading
            with torch_load_device("cpu") as patcher:
                assert patcher.device.type == "cpu"
                model = load_model()
            
            # Test GPU loading if available
            if torch.cuda.is_available():
                with torch_load_device("cuda:0") as patcher:
                    assert patcher.device.type == "cuda"
                    model = load_model()
    """
    def _make_patcher(device: Union[str, torch.device]):
        return TorchLoadPatcher(device)
    return _make_patcher


@pytest.fixture(params=["cpu", "cuda:0", "mps"])
def torch_load_all_devices(request):
    """Pytest fixture that runs tests on all available devices.
    
    This is useful for testing device-agnostic code on multiple devices.
    Tests using this fixture will be run once for each available device.
    
    Example:
        def test_on_all_devices(torch_load_all_devices):
            device_type = torch_load_all_devices.device.type
            if device_type == "cuda" and not torch.cuda.is_available():
                pytest.skip("CUDA not available")
            if device_type == "mps" and not torch.backends.mps.is_available():
                pytest.skip("MPS not available")
            
            model = load_model()
            # Test runs on each device
    """
    device = request.param
    
    # Skip if device not available
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip(f"CUDA not available")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip(f"MPS not available")
    
    patcher = TorchLoadPatcher(device)
    with patcher:
        yield patcher