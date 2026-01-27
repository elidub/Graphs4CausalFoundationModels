"""
DOFM baseline using PreprocessingGraphConditionedPFN wrapper.

This is a simplified version that uses the PreprocessingGraphConditionedPFN
wrapper which handles all preprocessing automatically.
"""

import argparse
import sys
import numpy as np

# Add paths
sys.path.insert(0, '<REPO_ROOT>')
sys.path.insert(0, '<REPO_ROOT>/RealCauseEval')

from src.models.PreprocessingGraphConditionedPFN import PreprocessingGraphConditionedPFN
from run_baselines.eval import evaluate_pipeline


def dofm_pipeline(model, cate_dataset):
    """
    Use the PreprocessingGraphConditionedPFN model to predict CATE.
    
    Args:
        model: PreprocessingGraphConditionedPFN instance (already loaded)
        cate_dataset: CATE_Dataset with X_train, t_train, y_train, X_test, true_cate
        
    Returns:
        cate_pred: Array of CATE predictions for X_test
    """
    # Extract data from cate_dataset
    X_train = cate_dataset.X_train
    t_train_orig = cate_dataset.t_train.reshape(-1, 1) if cate_dataset.t_train.ndim == 1 else cate_dataset.t_train
    y_train_orig = cate_dataset.y_train.reshape(-1, 1) if cate_dataset.y_train.ndim == 1 else cate_dataset.y_train
    X_test = cate_dataset.X_test
    y_train = y_train_orig

    # Print dataset info
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    n_features = X_train.shape[1]
    print(f"[Dataset] Train samples: {n_train}, Test samples: {n_test}, Features: {n_features}")

    # Target encoding for treatment: replace T with mean(Y|T)
    t_flat = t_train_orig.flatten()
    y_flat = y_train.flatten()
    mean_y_t0 = y_flat[t_flat == 0].mean()
    mean_y_t1 = y_flat[t_flat == 1].mean()
    t_train = np.where(t_train_orig == 0, mean_y_t0, mean_y_t1).astype(np.float32)
    
    # For intervention values, use the same target encoding
    t_intv_0_encoded = mean_y_t0
    t_intv_1_encoded = mean_y_t1
    
    print(f"[Target Encoding] T=0 -> {mean_y_t0:.4f}, T=1 -> {mean_y_t1:.4f}")
     
    n_test = X_test.shape[0]
    n_features_orig = X_train.shape[1]
    model_n_features = model.model.num_features
    
    # Fit preprocessing on training data
    model.fit(X_train, t_train, y_train)
    
    # Build adjacency matrix with known causal structure (partial graph format)
    # Shape: (model_n_features + 2, model_n_features + 2) for treatment + outcome + features
    # Partial graph format uses three states:
    #   -1: No edge/ancestor relationship (known non-edge)
    #   0: Unknown whether edge/ancestor exists
    #   1: Edge/ancestor relationship exists (known edge)
    
    # Position mapping (matching InterventionalDataset convention):
    # - Position 0: Treatment (T)
    # - Position 1: Outcome (Y)
    # - Positions 2 to model_n_features+1: Feature variables
    
    # Number of real (non-padded) features
    n_real_features = min(n_features_orig, model_n_features)
    
    # Initialize all as unknown (0)
    adjacency_matrix = np.zeros((model_n_features + 2, model_n_features + 2), dtype=np.float32)
    
    T_idx = 0
    Y_idx = 1
    feature_offset = 2  # Features start at position 2
    
    # 1. T -> Y (treatment causes outcome)
    adjacency_matrix[T_idx, Y_idx] = 0.0
    
    # 2. Real features -> T (features cause treatment)
    for i in range(n_real_features):
        adjacency_matrix[feature_offset + i, T_idx] = 0.0
    
    # 3. Real features -> Y (features cause outcome)
    for i in range(n_real_features):
        adjacency_matrix[feature_offset + i, Y_idx] = 0.0
    
    # Note: Relationships between real features remain unknown (0)
    
    # 4. PADDED features: Set all edges to -1 (no edge) since they don't exist
    for i in range(n_real_features, model_n_features):
        feat_idx = feature_offset + i
        adjacency_matrix[feat_idx, :] = -1.0
        adjacency_matrix[:, feat_idx] = -1.0
        adjacency_matrix[feat_idx, feat_idx] = -1.0
    
    # For CATE prediction, predict Y for both T=0 and T=1
    # CATE = E[Y|T=1,X] - E[Y|T=0,X]

    
    # Predict Y for T=1 (using target-encoded value)
    T_intv_1 = np.full((n_test, 1), t_intv_1_encoded, dtype=np.float32)
    y_pred_1 = model.predict(
        X_obs=X_train,
        T_obs=t_train,
        Y_obs=y_train,
        X_intv=X_test,
        T_intv=T_intv_1,
        adjacency_matrix=adjacency_matrix,
        prediction_type="mean",
        inverse_transform=True,  # Don't inverse transform yet
    )
    
    # Predict Y for T=0 (using target-encoded value)
    T_intv_0 = np.full((n_test, 1), t_intv_0_encoded, dtype=np.float32)
    y_pred_0 = model.predict(
        X_obs=X_train,
        T_obs=t_train,
        Y_obs=y_train,
        X_intv=X_test,
        T_intv=T_intv_0,
        adjacency_matrix=adjacency_matrix,
        prediction_type="mean",
        inverse_transform=True,
    )

    # CATE = E[Y|do(T=1)] - E[Y|do(T=0)]
    cate_pred = y_pred_1 - y_pred_0

    print(f"CATE predictions: {cate_pred.flatten()[:20]}...")
    
    return cate_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DOFM baseline with preprocessing wrapper.")

    parser.add_argument("--dataset", type=str, required=True, 
                        help="Name of the dataset (IHDP, ACIC, CPS, PSID) or 'all' for all datasets")
    parser.add_argument("--model", type=str, required=True, help="Model name for logging")    
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--checkpoint_path", type=str, default=None, 
                        help="Path to model checkpoint (default: final_earlytest_16773250.0)")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to model config (default: final_earlytest_16773250.0)")

    args = parser.parse_args()

    # Model paths - use defaults if not provided
    if args.checkpoint_path is None:
        checkpoint_path = "<REPO_ROOT>/experiments/FirstTests/checkpoints/final_earlytest_16773250.0/final_model_with_bardist.pt"
    else:
        checkpoint_path = args.checkpoint_path
        
    if args.config_path is None:
        model_config_path = "<REPO_ROOT>/experiments/FirstTests/checkpoints/final_earlytest_16773250.0/final_model_with_bardist_config.yaml"
    else:
        model_config_path = args.config_path
    
    print(f"Loading model from: {checkpoint_path}")
    model = PreprocessingGraphConditionedPFN(
        config_path=model_config_path,
        checkpoint_path=checkpoint_path,
        verbose=True,
    )
    model.load()

    # Determine which datasets to run
    ALL_DATASETS = ["IHDP", "ACIC", "CPS", "PSID"]
    if args.dataset.lower() == "all":
        datasets_to_run = ALL_DATASETS
    else:
        datasets_to_run = [args.dataset]
    
    # Run evaluation for each dataset
    for dataset in datasets_to_run:
        print(f"\n{'='*60}")
        print(f"--- Starting Experiment ---")
        print(f"Dataset: {dataset}")
        print(f"Model:   {args.model}")
        print(f"{'='*60}\n")
        
        # Create a copy of args with the current dataset
        dataset_args = argparse.Namespace(**vars(args))
        dataset_args.dataset = dataset
        
        try:
            evaluate_pipeline(
                exp_name=args.exp_name,
                model_pipeline=dofm_pipeline,
                model=model,
                args=dataset_args)
            print(f"\n--- {dataset} Experiment Finished ---")
        except Exception as e:
            print(f"\n--- {dataset} Experiment FAILED: {e} ---")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"--- All Experiments Finished ---")
    print(f"{'='*60}")
