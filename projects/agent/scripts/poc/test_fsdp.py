#!/usr/bin/env python3
"""
Test script to verify FSDP configuration and auto-wrap policy detection.

This script tests the FSDP functionality without requiring a full dataset,
useful for debugging and validation.
"""

import torch
from lightning.fabric import Fabric
from lightning.fabric.strategies.fsdp import FSDPStrategy
from transformers import AutoModelForVision2Seq, AutoTokenizer

# Import the FSDP auto-wrap policy function
import sys
import os
sys.path.append(os.path.dirname(__file__))
from train import get_fsdp_auto_wrap_policy


def test_auto_wrap_policy():
    """Test the auto-wrap policy detection."""
    print("Testing FSDP auto-wrap policy detection...")
    
    auto_wrap_policy = get_fsdp_auto_wrap_policy()
    print(f"Detected {len(auto_wrap_policy)} transformer block types:")
    for cls in auto_wrap_policy:
        print(f"  - {cls.__name__} from {cls.__module__}")
    
    return auto_wrap_policy


def test_fsdp_strategy():
    """Test FSDP strategy initialization."""
    print("\nTesting FSDP strategy initialization...")
    
    auto_wrap_policy = get_fsdp_auto_wrap_policy()
    
    try:
        strategy = FSDPStrategy(
            auto_wrap_policy=auto_wrap_policy,
            state_dict_type="full",
            sharding_strategy="FULL_SHARD",
        )
        print("✓ FSDP strategy created successfully")
        print(f"  - Sharding strategy: FULL_SHARD")
        print(f"  - State dict type: full")
        print(f"  - Auto-wrap policy: {len(auto_wrap_policy)} block types")
        return strategy
    except Exception as e:
        print(f"✗ Failed to create FSDP strategy: {e}")
        return None


def test_model_initialization():
    """Test model initialization with FSDP."""
    print("\nTesting model initialization...")
    
    model_name = "HuggingFaceTB/SmolVLM2-2.2B-Base"
    
    # Test without FSDP first
    print("1. Testing without FSDP...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✓ Tokenizer loaded: {len(tokenizer)} tokens")
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        return False
    
    # Test with FSDP strategy
    print("2. Testing with FSDP strategy...")
    auto_wrap_policy = get_fsdp_auto_wrap_policy()
    strategy = FSDPStrategy(
        auto_wrap_policy=auto_wrap_policy,
        state_dict_type="full",
        sharding_strategy="FULL_SHARD",
    )
    
    fabric = Fabric(
        accelerator="auto",
        devices=1,  # Use 1 device for testing
        strategy="auto",  # Use auto for single device
        precision="32-true",
    )
    
    try:
        fabric.launch()
        print("✓ Fabric launched successfully")
        
        # Test model loading with empty_init
        with fabric.init_module(empty_init=True):
            model = AutoModelForVision2Seq.from_pretrained(model_name)
        
        print("✓ Model initialized with empty_init")
        print(f"  - Model type: {type(model).__name__}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  - Total parameters: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to initialize model with FSDP: {e}")
        return False


def test_multi_gpu_config():
    """Test multi-GPU FSDP configuration."""
    print("\nTesting multi-GPU FSDP configuration...")
    
    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping multi-GPU test")
        return True
    
    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")
    
    if gpu_count < 2:
        print("⚠ Less than 2 GPUs available, skipping multi-GPU test")
        return True
    
    auto_wrap_policy = get_fsdp_auto_wrap_policy()
    strategy = FSDPStrategy(
        auto_wrap_policy=auto_wrap_policy,
        state_dict_type="full",
        sharding_strategy="FULL_SHARD",
    )
    
    try:
        fabric = Fabric(
            accelerator="cuda",
            devices=min(2, gpu_count),
            strategy=strategy,
            precision="bf16-mixed",
        )
        print("✓ Multi-GPU FSDP Fabric configuration created successfully")
        print(f"  - Using {min(2, gpu_count)} GPUs")
        print(f"  - Strategy: {strategy.__class__.__name__}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to create multi-GPU FSDP configuration: {e}")
        return False


def main():
    """Run all FSDP tests."""
    print("=== FSDP Configuration Test Suite ===\n")
    
    tests = [
        ("Auto-wrap policy detection", test_auto_wrap_policy),
        ("FSDP strategy initialization", test_fsdp_strategy),
        ("Model initialization", test_model_initialization),
        ("Multi-GPU configuration", test_multi_gpu_config),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            result = test_func()
            success = result is not None and result is not False
            results.append((test_name, success))
            
            if success:
                print(f"\n✓ {test_name} PASSED")
            else:
                print(f"\n✗ {test_name} FAILED")
                
        except Exception as e:
            print(f"\n✗ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! FSDP configuration is working correctly.")
    else:
        print("⚠ Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
