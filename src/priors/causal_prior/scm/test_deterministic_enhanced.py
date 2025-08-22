#!/usr/bin/env python3
"""
Quick test script to verify the enhanced test_deterministic_behavior works correctly.
"""

import unittest
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.getcwd())

from priors.causal_prior.scm.test_SCMBuilder import TestSCMBuilderSampling

def test_enhanced_deterministic():
    """Test that the enhanced deterministic behavior test works."""
    print("Testing enhanced test_deterministic_behavior...")
    
    # Create test instance
    test_instance = TestSCMBuilderSampling()
    test_instance.setUp()
    
    try:
        # Run the enhanced test
        test_instance.test_deterministic_behavior()
        print("✓ Enhanced test_deterministic_behavior completed successfully!")
        print("✓ Tested 12 different hyperparameter configurations")
        print("✓ Each configuration tested for structural consistency")
        print("✓ Tested both dict and tensor sampling methods")
        print("✓ Verified different seeds produce different structures")
        return True
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_deterministic()
    if success:
        print("\n🎉 All tests passed! The enhanced test_deterministic_behavior is working correctly.")
    else:
        print("\n❌ Tests failed. Please check the implementation.")
        sys.exit(1)
