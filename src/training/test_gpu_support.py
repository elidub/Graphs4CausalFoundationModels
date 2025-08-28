#!/usr/bin/env python3
"""
Test script to verify GPU support in simple_run.py
"""

import sys
import torch
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from training.simple_run import detect_device, setup_gpu_optimizations, cleanup_gpu_memory


def test_device_detection():
    """Test device detection functionality."""
    print("=== Testing Device Detection ===")
    
    # Test auto detection
    print("\n1. Auto detection:")
    device_auto = detect_device("auto", verbose=True)
    print(f"Selected device: {device_auto}")
    
    # Test explicit CPU
    print("\n2. Explicit CPU:")
    device_cpu = detect_device("cpu", verbose=True)
    print(f"Selected device: {device_cpu}")
    
    # Test explicit CUDA (if available)
    print("\n3. Explicit CUDA:")
    device_cuda = detect_device("cuda", verbose=True)
    print(f"Selected device: {device_cuda}")
    
    return device_auto


def test_gpu_optimizations():
    """Test GPU optimization setup."""
    print("\n=== Testing GPU Optimizations ===")
    
    # Test with CPU
    print("\n1. CPU optimizations:")
    dummy_model = torch.nn.Linear(10, 1)
    cpu_opts = setup_gpu_optimizations("cpu", dummy_model)
    print(f"CPU optimizations: {cpu_opts}")
    
    # Test with CUDA (if available)
    if torch.cuda.is_available():
        print("\n2. CUDA optimizations:")
        cuda_opts = setup_gpu_optimizations("cuda", dummy_model)
        print(f"CUDA optimizations: {cuda_opts}")
    else:
        print("\n2. CUDA not available - skipping CUDA optimization test")


def test_memory_cleanup():
    """Test GPU memory cleanup."""
    print("\n=== Testing Memory Cleanup ===")
    
    if torch.cuda.is_available():
        print("Before cleanup:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1e6:.1f} MB")
        
        # Create some tensors to use memory
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.randn(1000, 1000, device="cuda")
        z = torch.matmul(x, y)
        
        print("After creating tensors:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1e6:.1f} MB")
        
        # Cleanup
        del x, y, z
        cleanup_gpu_memory()
        
        print("After cleanup:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1e6:.1f} MB")
    else:
        print("CUDA not available - skipping memory cleanup test")


def main():
    """Run all GPU support tests."""
    print("GPU Support Test Script for simple_run.py")
    print("=" * 50)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    test_device_detection()
    test_gpu_optimizations()
    test_memory_cleanup()
    
    print("\n" + "=" * 50)
    print("GPU support tests completed!")


if __name__ == "__main__":
    main()
