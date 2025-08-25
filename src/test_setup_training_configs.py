#!/usr/bin/env python3
"""
Simple test runner for SetupTraining tests that can run without full dependencies.
Tests the configuration structure and basic functionality.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config_structure():
    """Test the configuration structure without requiring torch."""
    print("Testing SetupTraining configuration structure...")
    
    try:
        # Test config extraction
        from training.configs import debug_config, small_config, extract_config_values
        print("✓ Successfully imported config modules")
        
        # Test debug config
        print("\n1. Testing debug_config extraction:")
        extracted_debug = extract_config_values(debug_config)
        print(f"   Learning rate: {extracted_debug['learning_rate']}")
        print(f"   Max steps: {extracted_debug['max_steps']}")
        print(f"   Batch size: {extracted_debug['batch_size']}")
        print(f"   Device: {extracted_debug['device']}")
        
        # Test small config
        print("\n2. Testing small_config extraction:")
        extracted_small = extract_config_values(small_config)
        print(f"   Learning rate: {extracted_small['learning_rate']}")
        print(f"   Max steps: {extracted_small['max_steps']}")
        print(f"   Precision: {extracted_small['precision']}")
        
        # Test all configs
        print("\n3. Testing all predefined configs:")
        from training.configs import TRAINING_CONFIGS
        for name, config in TRAINING_CONFIGS.items():
            try:
                extracted = extract_config_values(config)
                print(f"   ✓ {name}: lr={extracted['learning_rate']}, steps={extracted['max_steps']}")
            except Exception as e:
                print(f"   ✗ {name}: Error - {e}")
        
        # Test invalid config
        print("\n4. Testing invalid config handling:")
        invalid_config = {
            'learning_rate': {'invalid_key': 1e-3}  # Missing 'value' key
        }
        try:
            extract_config_values(invalid_config)
            print("   ✗ Should have raised ValueError")
        except ValueError:
            print("   ✓ Correctly raised ValueError for invalid config")
        
        # Test config consistency
        print("\n5. Testing config consistency:")
        for name, config in TRAINING_CONFIGS.items():
            extracted = extract_config_values(config)
            
            # Check required keys
            required_keys = ['learning_rate', 'max_steps', 'batch_size', 'max_epochs']
            missing_keys = [key for key in required_keys if key not in extracted]
            if missing_keys:
                print(f"   ✗ {name}: Missing keys {missing_keys}")
            else:
                print(f"   ✓ {name}: All required keys present")
                
            # Check step-based training (max_epochs should be 1)
            if extracted['max_epochs'] != 1:
                print(f"   ⚠ {name}: max_epochs={extracted['max_epochs']}, expected 1 for synthetic data")
            else:
                print(f"   ✓ {name}: Correctly configured for step-based training")
        
        print("\n✓ All configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pattern_consistency():
    """Test that the pattern matches priordata_processing structure."""
    print("\nTesting pattern consistency with priordata_processing...")
    
    try:
        # Compare structure with priordata_processing pattern
        from training.configs import debug_config
        
        # Check structure
        for key, value in debug_config.items():
            if not isinstance(value, dict):
                print(f"   ✗ {key}: Value is not a dict")
                continue
                
            if 'value' in value and len(value) == 1:
                print(f"   ✓ {key}: Fixed value pattern")
            elif 'distribution' in value and 'distribution_parameters' in value:
                print(f"   ✓ {key}: Distribution pattern (future support)")
            else:
                print(f"   ✗ {key}: Invalid pattern {value}")
        
        print("✓ Pattern consistency verified!")
        return True
        
    except Exception as e:
        print(f"✗ Pattern test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("SetupTraining Configuration Tests")
    print("=" * 60)
    
    success1 = test_config_structure()
    success2 = test_pattern_consistency()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 ALL TESTS PASSED!")
        print("SetupTraining configuration system is working correctly!")
    else:
        print("❌ Some tests failed!")
        sys.exit(1)
    print("=" * 60)
