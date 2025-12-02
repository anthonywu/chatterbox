"""Examples of using the TorchLoadPatcher in various ways."""

import torch
from pathlib import Path
from chatterbox.utils.torch_load_patcher import (
    TorchLoadPatcher, 
    torch_load_with_device, 
    patch_torch_load
)


# Example 1: Using as a context manager with the class directly
def example_context_manager_class():
    """Example using TorchLoadPatcher class as context manager."""
    print("Example 1: Context manager with class")
    
    # Auto-detect best device
    with TorchLoadPatcher() as patcher:
        print(f"Using device: {patcher.device}")
        # All torch.load calls here will use the auto-detected device
        # model = torch.load("model.pt")  # Would use patcher.device
    
    # Specific device
    with TorchLoadPatcher("mps"):
        print("All torch.load calls now use MPS")
        # model = torch.load("model.pt")  # Would use MPS


# Example 2: Using the convenience context manager function
def example_context_manager_function():
    """Example using the convenience function."""
    print("\nExample 2: Context manager with convenience function")
    
    with torch_load_with_device("cpu"):
        print("Loading on CPU")
        # model = torch.load("model.pt")  # Would use CPU
    
    # Auto-detect
    with torch_load_with_device():
        print("Loading on auto-detected device")
        # model = torch.load("model.pt")


# Example 3: Using as a decorator on a function
@patch_torch_load("cpu")
def load_model_on_cpu(model_path: str):
    """This function will always load models on CPU."""
    print("\nExample 3: Decorator on function")
    print(f"Loading {model_path} on CPU")
    # return torch.load(model_path)  # Would use CPU
    return f"Model loaded from {model_path} on CPU"


# Example 4: Using decorator with auto-detection
@patch_torch_load()
def load_model_auto(model_path: str):
    """This function will load models on the best available device."""
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nExample 4: Decorator with auto-detection")
    print(f"Loading {model_path} on {device}")
    # return torch.load(model_path)  # Would use auto-detected device
    return f"Model loaded from {model_path} on {device}"


# Example 5: Using within a class method
class ModelLoader:
    def __init__(self, device: str = None):
        self.patcher = TorchLoadPatcher(device)
    
    def load_with_context(self, model_path: str):
        """Load model using context manager."""
        with self.patcher:
            print(f"\nExample 5: Class method with context")
            print(f"Loading {model_path} on {self.patcher.device}")
            # return torch.load(model_path)
            return f"Model loaded on {self.patcher.device}"
    
    @patch_torch_load("mps")
    def load_on_mps(self, model_path: str):
        """Always loads on MPS (if available)."""
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"\nExample 5b: Class method with decorator")
        print(f"Loading {model_path} on {device}")
        # return torch.load(model_path)
        return f"Model loaded on {device}"


# Example 6: Nested context managers
def example_nested_contexts():
    """Example showing nested context managers."""
    print("\nExample 6: Nested contexts")
    
    with torch_load_with_device("cpu"):
        print("Outer context: CPU")
        # model1 = torch.load("model1.pt")  # Uses CPU
        
        with torch_load_with_device("mps"):
            print("Inner context: MPS")
            # model2 = torch.load("model2.pt")  # Uses MPS
        
        print("Back to outer context: CPU")
        # model3 = torch.load("model3.pt")  # Uses CPU again


# Example 7: Real-world usage - updating the example_for_mac.py pattern
def example_mac_usage():
    """Example showing how to replace the hack in example_for_mac.py."""
    print("\nExample 7: Real-world Mac usage")
    
    from chatterbox.tts import ChatterboxTTS
    
    # Instead of the manual patching, just use the context manager
    with torch_load_with_device():  # Auto-detects MPS on Mac
        model = ChatterboxTTS.from_pretrained()
        print(f"Model loaded successfully")
        # Generate speech...


# Example 8: Using in pytest tests
def example_pytest_usage():
    """Example showing pytest usage patterns."""
    print("\nExample 8: Pytest patterns")
    print("""
    # In your test file:
    
    def test_model_loading(torch_load_patcher):
        '''Test that uses the auto-detecting fixture.'''
        model = MyModel.from_pretrained()
        assert model is not None
    
    def test_mps_specific(torch_load_with_mps):
        '''Test that specifically uses MPS.'''
        model = MyModel.from_pretrained()
        assert model.device.type in ["mps", "cpu"]
    
    def test_with_device_factory(torch_load_device):
        '''Test using the factory fixture.'''
        with torch_load_device("cuda"):
            model = load_cuda_model()
        
        with torch_load_device("cpu"):
            model = load_cpu_model()
    """)


# Run all examples
if __name__ == "__main__":
    print("TorchLoadPatcher Examples")
    print("=" * 50)
    
    example_context_manager_class()
    example_context_manager_function()
    
    # Function examples
    result = load_model_on_cpu("model.pt")
    print(f"Result: {result}")
    
    result = load_model_auto("model.pt")
    print(f"Result: {result}")
    
    # Class examples
    loader = ModelLoader()
    loader.load_with_context("model.pt")
    loader.load_on_mps("model.pt")
    
    # Nested contexts
    example_nested_contexts()
    
    # Real-world usage
    try:
        example_mac_usage()
    except Exception as e:
        print(f"Skipping real model load: {e}")
    
    # Pytest patterns
    example_pytest_usage()