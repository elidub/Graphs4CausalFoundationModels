#!/usr/bin/env python3
"""Simple test script to verify our fixes work."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import unittest
from training.Test_SetupTraining import TestSetupTraining

if __name__ == "__main__":
    # Create test suite with just the previously skipped tests
    suite = unittest.TestSuite()
    
    test_cases = [
        'test_get_methods',
        'test_model_setup_dict_config', 
        'test_model_setup_string_config',
        'test_quick_train_function',
        'test_save_model'
    ]
    
    for test_case in test_cases:
        suite.addTest(TestSetupTraining(test_case))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\nSummary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    print("\n✓ Test script completed!")
