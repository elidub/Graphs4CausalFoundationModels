"""
DOFM baseline with PSID-balanced sampling: all T=1 + up to 500 T=0.

Uses PreprocessingGraphConditionedPFN wrapper with balanced training sampling
to handle PSID's severe class imbalance.
"""

import argparse
import sys
import numpy as np

# Add paths
sys.path.insert(0, '/fast/arikreuter/DoPFN_v2/CausalPriorFitting')
sys.path.insert(0, '/fast/arikreuter/DoPFN_v2/CausalPriorFitting/RealCauseEval')

from src.models.PreprocessingGraphConditionedPFN import PreprocessingGraphConditionedPFN
from run_baselines.eval import evaluate_pipeline

# Global dataset name for pipeline
DATASET_NAME = None


def build_adjacency_matrix(model_n_features, n_real_features, graph_mode="full_graph"):
    """Build adjacency matrix based on graph knowledge mode."""
    adjacency_matrix = np.zeros((model_n_features + 2, model_n_features + 2), dtype=np.float32)
    
    T_idx = 0
    Y_idx = 1
    feature_offset = 2
    
    if graph_mode == "all_unknown":
        print(f"Graph knowledge: ALL UNKNOWN")
    elif graph_mode == "full_graph":
        adjacency_matrix[T_idx, Y_idx] = 1.0
        for i in range(n_real_features):
            adjacency_matrix[feature_offset + i, T_idx] = 1.0
            adjacency_matrix[feature_offset + i, Y_idx] = 1.0
        print(f"Graph knowledge: T->Y=1, X->T=1, X->Y=1 (full graph)")
    else:
        raise ValueError(f"Unknown graph_mode: {graph_mode}")
    
    # PADDED features: Set all edges to -1
    for i in range(n_real_features, model_n_features):
        feat_idx = feature_offset + i
        adjacency_matrix[feat_idx, :] = -1.0
        adjacency_matrix[:, feat_idx] = -1.0
        adjacency_matrix[feat_idx, feat_idx] = -1.0
    
    return adjacency_matrix


def create_dofm_psid_balanced_pipeline(graph_mode="full_graph"):
    """Factory to create pipeline with PSID-balanced sampling."""
    
    def dofm_psid_balanced_pipeline(model, cate_dataset):
        global DATASET_NAME
        
        # Extract data
        X_train_full = cate_dataset.X_train
        t_train_full = cate_dataset.t_train.reshape(-1, 1) if cate_dataset.t_train.ndim == 1 else cate_dataset.t_train
        y_train_full = cate_dataset.y_train.reshape(-1, 1) if cate_dataset.y_train.ndim == 1 else cate_dataset.y_train
        X_test = cate_dataset.X_test

        n_train_full = X_train_full.shape[0]
        n_test = X_test.shape[0]
        n_features = X_train_full.shape[1]

        # Default: use full training set
        X_train = X_train_full
        t_train_orig = t_train_full
        y_train = y_train_full

        # If PSID, subsample controls
        if DATASET_NAME is not None and DATASET_NAME.upper() == 'PSID':
            t_flat_full = t_train_full.flatten()
            treated_mask = (t_flat_full == 1)
            control_mask = (t_flat_full == 0)

            X_treated = X_train_full[treated_mask]
            t_treated = t_train_full[treated_mask]
            y_treated = y_train_full[treated_mask]

            X_control = X_train_full[control_mask]
            t_control = t_train_full[control_mask]
            y_control = y_train_full[control_mask]

            n_control = X_control.shape[0]
            n_keep_control = min(500, n_control)

            if n_control > n_keep_control:
                np.random.seed(42)
                indices = np.random.choice(n_control, n_keep_control, replace=False)
                X_control_sel = X_control[indices]
                t_control_sel = t_control[indices]
                y_control_sel = y_control[indices]
                print(f"[PSID Balanced] Kept all {X_treated.shape[0]} treated, sampled {n_keep_control}/{n_control} controls")
            else:
                X_control_sel = X_control
                t_control_sel = t_control
                y_control_sel = y_control
                print(f"[PSID Balanced] Kept all {X_treated.shape[0]} treated and all {n_control} controls")

            # Concatenate
            X_train = np.vstack([X_treated, X_control_sel])
            t_train_orig = np.vstack([t_treated, t_control_sel])
            y_train = np.vstack([y_treated, y_control_sel])

            # Shuffle
            perm = np.random.RandomState(42).permutation(X_train.shape[0])
            X_train = X_train[perm]
            t_train_orig = t_train_orig[perm]
            y_train = y_train[perm]

            print(f"[PSID Balanced] Final training size: {X_train.shape[0]}")
        else:
            print(f"[Dataset] Using full training set: {n_train_full} samples")

        n_train = X_train.shape[0]
        print(f"[Dataset] Train: {n_train}, Test: {n_test}, Features: {n_features}")

        # Target encoding on the sampled training set
        t_flat = t_train_orig.flatten()
        y_flat = y_train.flatten()
        mean_y_t0 = y_flat[t_flat == 0].mean() if np.any(t_flat == 0) else 0.0
        mean_y_t1 = y_flat[t_flat == 1].mean() if np.any(t_flat == 1) else 0.0
        t_train = np.where(t_train_orig == 0, mean_y_t0, mean_y_t1).astype(np.float32)
        
        t_intv_0_encoded = mean_y_t0
        t_intv_1_encoded = mean_y_t1
        
        print(f"[Target Encoding] T=0 -> {mean_y_t0:.4f}, T=1 -> {mean_y_t1:.4f}")
         
        n_features_orig = X_train.shape[1]
        model_n_features = model.model.num_features
        
        # Fit preprocessing
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
    
    return dofm_psid_balanced_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DOFM PSID-balanced baseline.")

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)    
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--graph_mode", type=str, default="full_graph", 
                        choices=["all_unknown", "full_graph"])

    args = parser.parse_args()
    
    DATASET_NAME = args.dataset
    graph_mode = args.graph_mode

    if args.checkpoint_path is None:
        checkpoint_path = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/checkpoints/final_earlytest_full_conditioning_16773252.0/final_model_with_bardist.pt"
    else:
        checkpoint_path = args.checkpoint_path
        
    if args.config_path is None:
        model_config_path = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/checkpoints/final_earlytest_full_conditioning_16773252.0/final_model_with_bardist_config.yaml"
    else:
        model_config_path = args.config_path
    
    print(f"Loading model from: {checkpoint_path}")
    print(f"Graph mode: {graph_mode}")
    print(f"PSID balanced sampling: all T=1 + up to 500 T=0")
    
    model = PreprocessingGraphConditionedPFN(
        config_path=model_config_path,
        checkpoint_path=checkpoint_path,
        verbose=True,
        use_clustering=False,  # No clustering since we're subsampling
    )
    model.load()

    dofm_pipeline = create_dofm_psid_balanced_pipeline(graph_mode)

    print(f"\n{'='*60}")
    print(f"--- Starting DOFM PSID-Balanced Experiment ---")
    print(f"Dataset: {args.dataset}, Graph: {graph_mode}")
    print(f"{'='*60}\n")
    
    try:
        evaluate_pipeline(
            exp_name=args.exp_name,
            model_pipeline=dofm_pipeline,
            model=model,
            args=args)
        print(f"\n--- Experiment Finished ---")
    except Exception as e:
        print(f"\n--- Experiment FAILED: {e} ---")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"--- Done ---")
    print(f"{'='*60}")
