"""
DOFM baseline using PreprocessingGraphConditionedPFN wrapper.

This is a simplified version that uses the PreprocessingGraphConditionedPFN
wrapper which handles all preprocessing automatically.
"""

import argparse
import sys
import numpy as np

# Add paths
sys.path.insert(0, '/Users/arikreuter/Documents/PhD/CausalPriorFitting')
sys.path.insert(0, '/Users/arikreuter/Documents/PhD/CausalPriorFitting/RealCauseEval')

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
    t_train = cate_dataset.t_train.reshape(-1, 1) if cate_dataset.t_train.ndim == 1 else cate_dataset.t_train
    y_train = cate_dataset.y_train.reshape(-1, 1) if cate_dataset.y_train.ndim == 1 else cate_dataset.y_train
    X_test = cate_dataset.X_test
    
    n_test = X_test.shape[0]
    n_features_orig = X_train.shape[1]
    model_n_features = model.model.num_features
    
    # Fit preprocessing on training data
    model.fit(X_train, t_train, y_train)
    
    # Build adjacency matrix with known causal structure (partial graph format)
    # Shape: (model_n_features + 2, model_n_features + 2) for features + treatment + outcome
    # Partial graph format uses three states:
    #   -1: No edge/ancestor relationship (known non-edge)
    #   0: Unknown whether edge/ancestor exists
    #   1: Edge/ancestor relationship exists (known edge)
    
    # Position mapping:
    # - Positions 0 to model_n_features-1: Feature variables
    # - Position model_n_features: Treatment (T)
    # - Position model_n_features+1: Outcome (Y)
    
    # Number of real (non-padded) features
    n_real_features = min(n_features_orig, model_n_features)
    
    # Initialize all as unknown (0)
    adjacency_matrix = np.zeros((model_n_features + 2, model_n_features + 2), dtype=np.float32)
    
    T_idx = model_n_features
    Y_idx = model_n_features + 1
    
    # 1. T -> Y (treatment causes outcome)
    adjacency_matrix[T_idx, Y_idx] = 1.0
    
    # 2. Real features -> T (features cause treatment)
    for i in range(n_real_features):
        adjacency_matrix[i, T_idx] = 1.0
    
    # 3. Real features -> Y (features cause outcome)
    for i in range(n_real_features):
        adjacency_matrix[i, Y_idx] = 1.0
    
    # Note: Relationships between real features remain unknown (0)
    
    # 4. PADDED features: Set all edges to -1 (no edge) since they don't exist
    for i in range(n_real_features, model_n_features):
        adjacency_matrix[i, :] = -1.0
        adjacency_matrix[:, i] = -1.0
        adjacency_matrix[i, i] = -1.0
    
    # For CATE prediction, predict Y for both T=0 and T=1
    # CATE = E[Y|T=1,X] - E[Y|T=0,X]
    
    # Predict Y for T=1
    T_intv_1 = np.ones((n_test, 1), dtype=np.float32)
    y_pred_1 = model.predict(
        X_obs=X_train,
        T_obs=t_train,
        Y_obs=y_train,
        X_intv=X_test,
        T_intv=T_intv_1,
        adjacency_matrix=adjacency_matrix,
        prediction_type="mean",
        inverse_transform=False,  # Don't inverse transform yet
    )
    
    # Predict Y for T=0
    T_intv_0 = np.zeros((n_test, 1), dtype=np.float32)
    y_pred_0 = model.predict(
        X_obs=X_train,
        T_obs=t_train,
        Y_obs=y_train,
        X_intv=X_test,
        T_intv=T_intv_0,
        adjacency_matrix=adjacency_matrix,
        prediction_type="mean",
        inverse_transform=False,  # Don't inverse transform yet
    )
    
    # CATE in scaled space
    cate_scaled = y_pred_1 - y_pred_0
    
    # Inverse transform CATE to original scale
    # For CATE (difference), we only scale by range/2
    cate_pred = model._inverse_transform_cate(cate_scaled)
    
    return cate_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DOFM baseline with preprocessing wrapper.")

    parser.add_argument("--dataset", type=str, required=True, 
                        help="Name of the dataset (IHDP, ACIC, CPS, PSID) or 'all' for all datasets")
    parser.add_argument("--model", type=str, required=True, help="Model name for logging")    
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")    

    args = parser.parse_args()

    # Model paths
    checkpoint_path = "/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments/FirstTests/checkpoints/final_earlytest_16773250.0/final_model_with_bardist.pt"
    model_config_path = "/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments/FirstTests/checkpoints/final_earlytest_16773250.0/final_model_with_bardist_config.yaml"
    
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
