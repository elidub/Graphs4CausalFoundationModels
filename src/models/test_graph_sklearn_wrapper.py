"""
Test script for GraphConditionedInterventionalPFNSklearn wrapper with all model types.
"""

import torch
import numpy as np
import yaml
import tempfile
import os
from pathlib import Path

# Import the wrapper
from GraphConditionedInterventionalPFN_sklearn import GraphConditionedInterventionalPFNSklearn

def test_model_type(mode: str, checkpoint_path: str = None):
    """Test a specific graph conditioning mode."""
    print("\n" + "="*80)
    print(f"Testing: {mode}")
    print("="*80)
    
    # Create temporary config
    config = {
        'model_config': {
            'num_features': 3,
            'd_model': 64,
            'depth': 2,
            'heads_feat': 4,
            'heads_samp': 4,
            'dropout': 0.0,
            'hidden_mult': 2,
            'normalize_features': True,
            'n_sample_attention_sink_rows': 0,
            'n_feature_attention_sink_cols': 0,
            'graph_conditioning_mode': mode,
            'use_bar_distribution': False,  # Disable for testing without checkpoint
            'output_dim': 1,
        }
    }
    
    # Add mode-specific parameters
    if mode == 'soft_learned_bias':
        config['model_config']['graph_bias_init'] = -5.0
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        # Create wrapper
        wrapper = GraphConditionedInterventionalPFNSklearn(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device="cpu",
            verbose=True,
        )
        
        # Load model
        wrapper.load()
        print(f"✓ Model loaded successfully")
        print(f"  Model type: {type(wrapper.model).__name__}")
        print(f"  Graph conditioning mode: {wrapper.graph_conditioning_mode}")
        
        # Create synthetic data
        num_features = 3
        N, M = 20, 5
        
        X_obs = np.random.randn(N, num_features).astype(np.float32)
        T_obs = np.random.randn(N, 1).astype(np.float32)
        Y_obs = np.random.randn(N).astype(np.float32)
        X_intv = np.random.randn(M, num_features).astype(np.float32)
        T_intv = np.random.randn(M, 1).astype(np.float32)
        
        # Create adjacency matrix (L+2 = 5)
        adjacency_matrix = np.ones((num_features + 2, num_features + 2), dtype=np.float32)
        
        print(f"\n[Test Predictions]")
        print(f"  Data shapes: X_obs={X_obs.shape}, T_obs={T_obs.shape}, Y_obs={Y_obs.shape}")
        print(f"  Adjacency matrix: {adjacency_matrix.shape}")
        
        # Test point predictions (direct output, no BarDistribution)
        preds_point = wrapper.predict(
            X_obs, T_obs, Y_obs, X_intv, T_intv, adjacency_matrix, 
            prediction_type="point"
        )
        print(f"✓ Point predictions: shape={preds_point.shape}, sample={preds_point[:3]}")
        
        print(f"\n✓ All tests passed for {mode}!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if os.path.exists(config_path):
            os.unlink(config_path)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("GraphConditionedInterventionalPFNSklearn Test Suite")
    print("Testing all graph conditioning modes")
    print("="*80)
    
    modes = [
        'hard_attention_only',
        'soft_learned_bias',
        'hybrid_half_and_half',
        'flat_append'
    ]
    
    results = {}
    for mode in modes:
        results[mode] = test_model_type(mode)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for mode, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {mode:30s} {status}")
    
    print("\n" + "="*80)
    all_passed = all(results.values())
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*80)
