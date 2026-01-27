"""
DOFM Full Conditioning baseline using PreprocessingGraphConditionedPFN wrapper.

This uses the full conditioning model (test_feature_mask_fraction=0.0)
with the PreprocessingGraphConditionedPFN wrapper for automatic preprocessing.

Supports different graph knowledge modes:
- all_unknown: No graph knowledge (all edges unknown)
- t_to_y_only: Only T->Y=1 known
- x_to_t_only: X->T=1 and T->Y=1 known, X->Y unknown
- x_to_y_only: X->Y=1 and T->Y=1 known, X->T unknown
- full_graph: All edges known (X->T=1, X->Y=1, T->Y=1)
"""

import argparse
import sys
import numpy as np

# Add paths
sys.path.insert(0, '<REPO_ROOT>')
sys.path.insert(0, '<REPO_ROOT>/RealCauseEval')

from src.models.PreprocessingGraphConditionedPFN import PreprocessingGraphConditionedPFN
from run_baselines.eval import evaluate_pipeline


def build_adjacency_matrix(model_n_features, n_real_features, graph_mode="full_graph"):
    """Build adjacency matrix based on graph knowledge mode."""
    # Initialize all as unknown (0)
    adjacency_matrix = np.zeros((model_n_features + 2, model_n_features + 2), dtype=np.float32)
    
    T_idx = 0
    Y_idx = 1
    feature_offset = 2  # Features start at position 2
    
    if graph_mode == "all_unknown":
        print(f"Graph knowledge: ALL UNKNOWN (no graph information provided)")
    elif graph_mode == "t_to_y_only":
        adjacency_matrix[T_idx, Y_idx] = 1.0
        print(f"Graph knowledge: T->Y=1 only")
    elif graph_mode == "x_to_t_only":
        adjacency_matrix[T_idx, Y_idx] = 1.0
        for i in range(n_real_features):
            adjacency_matrix[feature_offset + i, T_idx] = 1.0
        print(f"Graph knowledge: T->Y=1, X->T=1, X->Y=0")
    elif graph_mode == "x_to_y_only":
        adjacency_matrix[T_idx, Y_idx] = 1.0
        for i in range(n_real_features):
            adjacency_matrix[feature_offset + i, Y_idx] = 1.0
        print(f"Graph knowledge: T->Y=1, X->T=0, X->Y=1")
    elif graph_mode == "full_graph":
        adjacency_matrix[T_idx, Y_idx] = 1.0
        for i in range(n_real_features):
            adjacency_matrix[feature_offset + i, T_idx] = 1.0
            adjacency_matrix[feature_offset + i, Y_idx] = 1.0
        print(f"Graph knowledge: T->Y=1, X->T=1, X->Y=1 (full graph)")
    else:
        raise ValueError(f"Unknown graph_mode: {graph_mode}")
    
    # PADDED features: Set all edges to -1 (no edge)
    for i in range(n_real_features, model_n_features):
        feat_idx = feature_offset + i
        adjacency_matrix[feat_idx, :] = -1.0
        adjacency_matrix[:, feat_idx] = -1.0
        adjacency_matrix[feat_idx, feat_idx] = -1.0
    
    return adjacency_matrix


def create_dofm_full_conditioning_pipeline(graph_mode="full_graph"):
    """Factory function to create a pipeline with specific graph mode."""
    def dofm_full_conditioning_pipeline(model, cate_dataset):
        # Extract data from cate_dataset
        X_train = cate_dataset.X_train
        t_train_orig = cate_dataset.t_train.reshape(-1, 1) if cate_dataset.t_train.ndim == 1 else cate_dataset.t_train
        y_train_orig = cate_dataset.y_train.reshape(-1, 1) if cate_dataset.y_train.ndim == 1 else cate_dataset.y_train
        X_test = cate_dataset.X_test
        y_train = y_train_orig

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
        
        t_intv_0_encoded = mean_y_t0
        t_intv_1_encoded = mean_y_t1
        
        print(f"[Target Encoding] T=0 -> {mean_y_t0:.4f}, T=1 -> {mean_y_t1:.4f}")
         
        n_test = X_test.shape[0]
        n_features_orig = X_train.shape[1]
        model_n_features = model.model.num_features
        
        model.fit(X_train, t_train, y_train)
        
        n_real_features = min(n_features_orig, model_n_features)
        adjacency_matrix = build_adjacency_matrix(model_n_features, n_real_features, graph_mode)
        
        # Predict Y for T=1
        T_intv_1 = np.full((n_test, 1), t_intv_1_encoded, dtype=np.float32)
        y_pred_1 = model.predict(
            X_obs=X_train, T_obs=t_train, Y_obs=y_train,
            X_intv=X_test, T_intv=T_intv_1,
            adjacency_matrix=adjacency_matrix,
            prediction_type="mean", inverse_transform=True,
        )
        
        # Predict Y for T=0
        T_intv_0 = np.full((n_test, 1), t_intv_0_encoded, dtype=np.float32)
        y_pred_0 = model.predict(
            X_obs=X_train, T_obs=t_train, Y_obs=y_train,
            X_intv=X_test, T_intv=T_intv_0,
            adjacency_matrix=adjacency_matrix,
            prediction_type="mean", inverse_transform=True,
        )

        cate_pred = y_pred_1 - y_pred_0
        print(f"CATE predictions: {cate_pred.flatten()[:20]}...")
        
        return cate_pred
    
    return dofm_full_conditioning_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DOFM full conditioning baseline.")

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)    
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    
    parser.add_argument("--all_unknown", action="store_true")
    parser.add_argument("--t_to_y_only", action="store_true")
    parser.add_argument("--x_to_t_only", action="store_true")
    parser.add_argument("--x_to_y_only", action="store_true")

    args = parser.parse_args()

    if args.all_unknown:
        graph_mode = "all_unknown"
    elif args.t_to_y_only:
        graph_mode = "t_to_y_only"
    elif args.x_to_t_only:
        graph_mode = "x_to_t_only"
    elif args.x_to_y_only:
        graph_mode = "x_to_y_only"
    else:
        graph_mode = "full_graph"

    if args.checkpoint_path is None:
        checkpoint_path = "<REPO_ROOT>/experiments/FirstTests/checkpoints/final_earlytest_full_conditioning_16773252.0/final_model_with_bardist.pt"
    else:
        checkpoint_path = args.checkpoint_path
        
    if args.config_path is None:
        model_config_path = "<REPO_ROOT>/experiments/FirstTests/checkpoints/final_earlytest_full_conditioning_16773252.0/final_model_with_bardist_config.yaml"
    else:
        model_config_path = args.config_path
    
    print(f"Loading full conditioning model from: {checkpoint_path}")
    print(f"Graph mode: {graph_mode}")
    
    model = PreprocessingGraphConditionedPFN(
        config_path=model_config_path,
        checkpoint_path=checkpoint_path,
        verbose=True,
    )
    model.load()

    dofm_pipeline = create_dofm_full_conditioning_pipeline(graph_mode)

    ALL_DATASETS = ["IHDP", "ACIC", "CPS", "PSID"]
    if args.dataset.lower() == "all":
        datasets_to_run = ALL_DATASETS
    else:
        datasets_to_run = [args.dataset]
    
    for dataset in datasets_to_run:
        print(f"\n{'='*60}")
        print(f"--- Starting Full Conditioning Experiment ---")
        print(f"Dataset: {dataset}, Model: {args.model}, Graph: {graph_mode}")
        print(f"{'='*60}\n")
        
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
