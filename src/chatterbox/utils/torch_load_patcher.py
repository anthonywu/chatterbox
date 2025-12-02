"""Thread-safe context manager and decorator for patching torch.load with device mapping.

WARNING: This module uses global state modification (monkey-patching) which has inherent
limitations with thread safety. While we use thread-local storage to minimize issues,
concurrent usage across threads may still have edge cases. For production multi-threaded
applications, consider alternative approaches like explicit device mapping.
"""

import torch
import functools
import threading
import warnings
from contextlib import contextmanager
from typing import Optional, Union, Callable, Any


# Thread-local storage for managing patches per thread
_thread_local = threading.local()


def _get_thread_patch_stack():
    """Get the patch stack for the current thread."""
    if not hasattr(_thread_local, 'patch_stack'):
        _thread_local.patch_stack = []
    return _thread_local.patch_stack


class TorchLoadPatcher:
    """Context manager and decorator for patching torch.load to use a specific device map_location.
    
    This is useful for ensuring all torch.load calls use the correct device (e.g., MPS on Mac,
    CUDA on GPU machines) without having to modify every torch.load call in the codebase.
    
    Thread Safety Note: This patcher uses thread-local storage to maintain separate patch
    stacks per thread. However, the underlying torch.load is still a global function,
    so there may be edge cases in heavily concurrent scenarios.
    """
    
    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        """Initialize the patcher with a specific device.
        
        Args:
            device: Device to use for map_location. If None, will auto-detect best available.
                   For CUDA, can specify index like "cuda:0" or "cuda:1".
        """
        if device is None:
            device = self._auto_detect_device()
        
        # Validate device
        try:
            self.device = torch.device(device) if isinstance(device, str) else device
        except RuntimeError as e:
            raise ValueError(f"Invalid device '{device}': {e}")
        
        self._original_torch_load = None
        self._is_active = False
    
    @staticmethod
    def _auto_detect_device() -> str:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            # Default to cuda:0 for multi-GPU systems
            return "cuda:0"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _patched_torch_load(self, *args, **kwargs) -> Any:
        """Patched version of torch.load that adds map_location if not specified."""
        # Get the topmost patcher from the thread-local stack
        patch_stack = _get_thread_patch_stack()
        if patch_stack:
            current_patcher = patch_stack[-1]
            if 'map_location' not in kwargs:
                kwargs['map_location'] = current_patcher.device
            return current_patcher._original_torch_load(*args, **kwargs)
        else:
            # Fallback to original if no patcher active (shouldn't happen)
            return self._original_torch_load(*args, **kwargs)
    
    def __enter__(self):
        """Enter the context manager and patch torch.load."""
        if self._is_active:
            raise RuntimeError(
                "Cannot re-enter the same TorchLoadPatcher instance. "
                "Create a new instance for nested contexts."
            )
        
        self._is_active = True
        patch_stack = _get_thread_patch_stack()
        
        # Save the current torch.load (might be original or already patched)
        self._original_torch_load = torch.load
        
        # Add ourselves to the stack
        patch_stack.append(self)
        
        # Always patch with our method (it will use the stack to determine behavior)
        torch.load = self._patched_torch_load
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and restore torch.load."""
        patch_stack = _get_thread_patch_stack()
        
        # Remove ourselves from the stack
        if patch_stack and patch_stack[-1] is self:
            patch_stack.pop()
        
        # Restore the previous torch.load
        if self._original_torch_load is not None:
            torch.load = self._original_torch_load
            self._original_torch_load = None
        
        self._is_active = False
        return False
    
    def __call__(self, func: Callable) -> Callable:
        """Use as a decorator for functions."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a new instance for each call to avoid re-entrance issues
            with TorchLoadPatcher(self.device):
                return func(*args, **kwargs)
        return wrapper


# Convenience function for context manager usage
@contextmanager
def torch_load_with_device(device: Optional[Union[str, torch.device]] = None):
    """Context manager for patching torch.load with a specific device.
    
    Args:
        device: Device to use for map_location. If None, will auto-detect best available.
                For CUDA, can specify index like "cuda:0" or "cuda:1".
    
    Example:
        with torch_load_with_device("mps"):
            model = MyModel.from_pretrained()  # All torch.load calls will use mps
        
        # Nested contexts work correctly
        with torch_load_with_device("cpu"):
            # loads on CPU
            with torch_load_with_device("cuda:0"):
                # loads on CUDA GPU 0
            # back to CPU
    """
    patcher = TorchLoadPatcher(device)
    with patcher:
        yield patcher


# Decorator version
def patch_torch_load(device: Optional[Union[str, torch.device]] = None):
    """Decorator for patching torch.load within a function.
    
    Args:
        device: Device to use for map_location. If None, will auto-detect best available.
                For CUDA, can specify index like "cuda:0" or "cuda:1".
    
    Example:
        @patch_torch_load("mps")
        def load_model():
            return torch.load("model.pt")  # Will use mps device
        
        @patch_torch_load("cuda:1")  # Use second GPU
        def load_on_gpu1():
            return torch.load("model.pt")
    """
    return TorchLoadPatcher(device)