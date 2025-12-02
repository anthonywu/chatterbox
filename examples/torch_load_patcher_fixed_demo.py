"""Demo showing the fixes for torch_load_patcher blockers."""

import torch
import threading
import time
from chatterbox.utils.torch_load_patcher import torch_load_with_device, patch_torch_load


def demo_nested_contexts():
    """Demonstrate that nested contexts now work correctly."""
    print("=== Nested Contexts Demo ===")
    
    print("Original torch.load:", torch.load)
    
    with torch_load_with_device("cpu"):
        print("Level 1 (CPU): torch.load is patched")
        
        with torch_load_with_device("mps" if torch.backends.mps.is_available() else "cpu"):
            print("Level 2 (MPS/CPU): torch.load is patched differently")
        
        print("Back to Level 1 (CPU): torch.load is restored to CPU patcher")
    
    print("Outside all contexts: torch.load is fully restored")
    print("Final torch.load:", torch.load)
    print()


def demo_thread_safety():
    """Demonstrate thread-safe operation."""
    print("=== Thread Safety Demo ===")
    
    results = {}
    
    def thread_func(thread_id, device):
        with torch_load_with_device(device) as patcher:
            results[thread_id] = {
                'device': patcher.device.type,
                'thread_id': threading.current_thread().ident
            }
            time.sleep(0.1)  # Simulate work
    
    # Start multiple threads with different devices
    threads = []
    devices = ["cpu", "cuda:0" if torch.cuda.is_available() else "cpu", "mps" if torch.backends.mps.is_available() else "cpu"]
    
    for i, device in enumerate(devices):
        t = threading.Thread(target=thread_func, args=(f"thread_{i}", device))
        threads.append(t)
        t.start()
    
    # Wait for all threads
    for t in threads:
        t.join()
    
    # Show results
    for thread_id, info in results.items():
        print(f"{thread_id}: device={info['device']}, thread={info['thread_id']}")
    print()


def demo_cuda_index_support():
    """Demonstrate CUDA device index support."""
    print("=== CUDA Index Support Demo ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping demo")
        return
    
    # Auto-detection now returns cuda:0
    with torch_load_with_device() as patcher:
        print(f"Auto-detected device: {patcher.device}")
    
    # Explicit CUDA indices
    for i in range(min(2, torch.cuda.device_count())):
        with torch_load_with_device(f"cuda:{i}") as patcher:
            print(f"Using CUDA device {i}: {patcher.device}")
    print()


def demo_error_handling():
    """Demonstrate improved error handling."""
    print("=== Error Handling Demo ===")
    
    # Invalid device
    try:
        with torch_load_with_device("invalid_gpu"):
            pass
    except ValueError as e:
        print(f"Caught expected error for invalid device: {e}")
    
    # Re-entrance protection
    patcher = torch_load_with_device("cpu").__enter__()
    try:
        patcher.__enter__()  # Try to re-enter
    except RuntimeError as e:
        print(f"Caught expected error for re-entrance: {e}")
    finally:
        patcher.__exit__(None, None, None)
    print()


@patch_torch_load("cpu")
def demo_decorator_usage():
    """Demonstrate decorator with proper instance creation."""
    print("=== Decorator Demo ===")
    print("This function always loads on CPU due to decorator")
    # Each call creates a new patcher instance, avoiding re-entrance issues
    return "Loaded successfully"


if __name__ == "__main__":
    print("TorchLoadPatcher - Fixed Version Demo")
    print("=" * 50)
    print()
    
    demo_nested_contexts()
    demo_thread_safety()
    demo_cuda_index_support()
    demo_error_handling()
    
    # Decorator demo
    result = demo_decorator_usage()
    print(f"Decorator result: {result}")
    
    # Multiple decorator calls work fine now
    for i in range(3):
        result = demo_decorator_usage()
        print(f"Call {i+1}: {result}")
    
    print("\n✅ All demos completed successfully!")