#!/usr/bin/env python3
"""
Test script for the new training configuration structure.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config_structure():
    """Test the new configuration structure."""
    print("Testing new training configuration structure...")
    
    try:
        from training.configs import debug_config, extract_config_values
        print("✓ Successfully imported config modules")
        
        # Test the structure
        print("\n1. Testing config structure:")
        print(f"   learning_rate config: {debug_config['learning_rate']}")
        print(f"   batch_size config: {debug_config['batch_size']}")
        
        # Test extraction
        print("\n2. Testing value extraction:")
        extracted = extract_config_values(debug_config)
        print(f"   Extracted learning_rate: {extracted['learning_rate']}")
        print(f"   Extracted batch_size: {extracted['batch_size']}")
        
        # Test all configs
        print("\n3. Testing all configs:")
        from training.configs import TRAINING_CONFIGS
        for name, config in TRAINING_CONFIGS.items():
            extracted = extract_config_values(config)
            print(f"   {name}: lr={extracted['learning_rate']}, steps={extracted['max_steps']}")
        
        print("\n✓ All tests passed! Configuration structure is working correctly.")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_config_structure()
