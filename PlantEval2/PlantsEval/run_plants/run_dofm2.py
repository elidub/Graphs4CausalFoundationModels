"""
DOFM for Plant CID benchmark (PlantEval2) using PreprocessingGraphConditionedPFN wrapper.

PlantEval2 version:
- Clean dataset with 6 features (X, Y, Bed_Position, Red/Blue/Far_Red Light)
- Treatment: Light_Intensity_umols
- Outcome: Biomass_g
- Adjacency matrix included in dataset
"""

import argparse
import sys
import os
import numpy as np
import torch

# Add paths for imports - use relative path first for local eval.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # Current directory first
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/fast/arikreuter/DoPFN_v2/CausalPriorFitting')
sys.path.insert(0, '/fast/arikreuter/DoPFN_v2/CausalPriorFitting/PlantEval2/PlantsEval')

from eval import evaluate_pipeline
from src.models.PreprocessingGraphConditionedPFN import PreprocessingGraphConditionedPFN
from src.utils.graph_utils import adjacency_to_ancestor_matrix


# PlantEval2 feature names (6 features)
# X_train columns: X, Y, Bed_Position, Red_Light_Intensity_umols, 
#                  Blue_Light_Intensity_umols, Far_Red_Light_Intensity_umols
FEATURE_NAMES = [
    "X",
    "Y", 
    "Bed_Position",
    "Red_Light_Intensity_umols",
    "Blue_Light_Intensity_umols",
    "Far_Red_Light_Intensity_umols"
]


def prepare_adjacency_matrix_for_model(dataset_adjacency, model_n_features, n_real_features, graph_mode="full_graph"):
    """
    Prepare adjacency matrix from dataset for the model.
    
    The dataset adjacency is 8x8: [features (6), treatment (1), outcome (1)]
    The model needs: [treatment (0), outcome (1), features (2-7), padding...]
    
    Args:
        dataset_adjacency: Adjacency matrix from dataset (8x8)
        model_n_features: Number of features the model expects
        n_real_features: Actual number of features in data (6)
        graph_mode: How to use the graph ("full_graph", "all_unknown", "t_to_y_only")
    
    Returns:
        Adjacency matrix in model's expected format
    """
    
    if graph_mode == "all_unknown":
        # Ignore the dataset graph, return all zeros (unknown) for real features
        adjacency_matrix = np.zeros((model_n_features + 2, model_n_features + 2), dtype=np.float32)
        
        # Pad non-existent features with -1 (no edge) since they don't exist
        if model_n_features > n_real_features:
            for i in range(n_real_features, model_n_features):
                feat_idx_pad = 2 + i
                adjacency_matrix[feat_idx_pad, :] = -1.0
                adjacency_matrix[:, feat_idx_pad] = -1.0
        
        print(f"Graph knowledge: ALL UNKNOWN (ignoring dataset graph)")
        print(f"  Real features: {n_real_features}, Padded features: {model_n_features - n_real_features}")
        return adjacency_matrix
    
    elif graph_mode == "t_to_y_only":
        # Only T -> Y edge, rest unknown for real features
        adjacency_matrix = np.zeros((model_n_features + 2, model_n_features + 2), dtype=np.float32)
        adjacency_matrix[0, 1] = 1.0  # T (idx 0) -> Y (idx 1)
        
        # Pad non-existent features with -1 (no edge) since they don't exist
        if model_n_features > n_real_features:
            for i in range(n_real_features, model_n_features):
                feat_idx_pad = 2 + i
                adjacency_matrix[feat_idx_pad, :] = -1.0
                adjacency_matrix[:, feat_idx_pad] = -1.0
        
        print(f"Graph knowledge: T->Y ONLY (ignoring dataset graph)")
        print(f"  Real features: {n_real_features}, Padded features: {model_n_features - n_real_features}")
        return adjacency_matrix
    
    elif graph_mode == "full_graph":
        # Use the dataset adjacency matrix
        if dataset_adjacency is None:
            raise ValueError("graph_mode='full_graph' requires dataset to have adjacency_matrix attribute")
        
        print(f"Using adjacency matrix from dataset")
        print(f"  Dataset adjacency shape: {dataset_adjacency.shape}")
        print(f"  Edges in dataset adjacency: {(dataset_adjacency == 1).sum()}")
        
        n_dataset_vars = dataset_adjacency.shape[0]
        
        # Dataset order: features (0-5), treatment (6), outcome (7)
        # Model order: treatment (0), outcome (1), features (2-7)
        dataset_to_model = np.zeros(n_dataset_vars, dtype=int)
        dataset_to_model[6] = 0  # treatment: dataset idx 6 -> model idx 0
        dataset_to_model[7] = 1  # outcome: dataset idx 7 -> model idx 1
        for i in range(6):  # features: dataset idx 0-5 -> model idx 2-7
            dataset_to_model[i] = 2 + i
        
        # Create model adjacency matrix with padding
        model_adjacency = np.zeros((model_n_features + 2, model_n_features + 2), dtype=np.float32)
        
        # Copy edges from dataset adjacency to model adjacency with reordering
        for i in range(n_dataset_vars):
            for j in range(n_dataset_vars):
                model_i = dataset_to_model[i]
                model_j = dataset_to_model[j]
                model_adjacency[model_i, model_j] = dataset_adjacency[i, j]

        # Convert adjacency matrix to ancestor matrix (transitive closure)
        model_adjacency_tensor = torch.tensor(model_adjacency, dtype=torch.float32)
        ancestor_matrix = adjacency_to_ancestor_matrix(model_adjacency_tensor, assume_dag=True)
        model_adjacency = ancestor_matrix.numpy()
        
        print(f"  Converted adjacency to ancestor matrix")
        print(f"  Ancestor relationships: {(model_adjacency == 1).sum()}")
        
        # Pad non-existent features with -1 (no edge) since they don't exist
        # These padded features are not real variables, so they have no edges
        if model_n_features > n_real_features:
            for i in range(n_real_features, model_n_features):
                feat_idx_pad = 2 + i
                model_adjacency[feat_idx_pad, :] = -1.0
                model_adjacency[:, feat_idx_pad] = -1.0
        
        print(f"  Model adjacency shape: {model_adjacency.shape}")
        print(f"  Edges in model adjacency: {(model_adjacency == 1).sum()}")

        # con
        
        return model_adjacency
    
    else:
        raise ValueError(f"Unknown graph_mode: {graph_mode}. Use 'all_unknown', 't_to_y_only', or 'full_graph'")



def create_dofm_plant_pipeline(graph_mode="full_graph"):
    """Factory function to create a pipeline with specific graph mode."""
    
    def dofm_plant_pipeline(model, cid_dataset):
        """
        Use PreprocessingGraphConditionedPFN to predict dose-response (CID).
        
        PlantEval2: Features are already clean, no encoding needed.
        """
        # Extract data - already clean
        X_train = np.array(cid_dataset.X_train, dtype=np.float32)
        X_test = np.array(cid_dataset.X_test, dtype=np.float32)
        
        t_train = np.array(cid_dataset.t_train, dtype=np.float32).reshape(-1, 1)
        y_train = np.array(cid_dataset.y_train, dtype=np.float32).reshape(-1, 1)
        t_test = np.array(cid_dataset.t_test, dtype=np.float32).reshape(-1, 1)
        
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        n_features = X_train.shape[1]
        print(f"[Dataset] Train: {n_train}, Test: {n_test}, Features: {n_features}")
        
        # Get model's expected number of features
        model_n_features = model.model.num_features
        n_features_orig = n_features
        
        # Fit preprocessing on training data
        model.fit(X_train, t_train, y_train)
        
        # Try to get adjacency matrix from dataset
        dataset_adjacency = None
        if hasattr(cid_dataset, 'adjacency_matrix') and cid_dataset.adjacency_matrix is not None:
            dataset_adjacency = cid_dataset.adjacency_matrix
            print(f"[Graph] Found adjacency matrix in dataset: {dataset_adjacency.shape}")
        
        if graph_mode == "full_graph" and dataset_adjacency is None:
            raise ValueError("graph_mode='full_graph' requires dataset to have adjacency_matrix attribute")
        
        # Prepare adjacency matrix for the model
        # This will be automatically converted to ancestor matrix if needed by PreprocessingGraphConditionedPFN
        n_real_features = min(n_features_orig, model_n_features)
        adjacency_matrix = prepare_adjacency_matrix_for_model(
            dataset_adjacency, model_n_features, n_real_features, graph_mode)
        
        # Predict Y for each test sample at its intervention value t_test
        # The model will automatically convert adjacency -> ancestor matrix if needed
        cid_pred = model.predict(
            X_obs=X_train,
            T_obs=t_train,
            Y_obs=y_train,
            X_intv=X_test,
            T_intv=t_test,
            adjacency_matrix=adjacency_matrix,
            prediction_type="mean",
            inverse_transform=True,
        )
        
        return cid_pred.flatten()
    
    return dofm_plant_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DOFM on plant CID benchmark (simplified).")

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)    
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--graph_mode", type=str, default="full_graph",
                        choices=["all_unknown", "t_to_y_only", "full_graph", "true_graph"])

    args = parser.parse_args()
    
    graph_mode = args.graph_mode

    if args.checkpoint_path is None:
        checkpoint_path = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/checkpoints/final_earlytest_full_conditioning_16773252.0/final_model_with_bardist.pt"
    else:
        checkpoint_path = args.checkpoint_path
        
    if args.config_path is None:
        model_config_path = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/checkpoints/final_earlytest_full_conditioning_16773252.0/final_model_with_bardist_config.yaml"
    else:
        model_config_path = args.config_path
    
    print(f"--- Starting DOFM Plant Experiment (PlantEval2) ---")
    print(f"Dataset: {args.dataset}")
    print(f"Model:   {args.model}")
    print(f"Graph mode: {graph_mode}")
    print(f"Loading model from: {checkpoint_path}")
    
    model = PreprocessingGraphConditionedPFN(
        config_path=model_config_path,
        checkpoint_path=checkpoint_path,
        verbose=True,
        use_clustering=False,  # No clustering for plant data
    )
    model.load()

    dofm_pipeline = create_dofm_plant_pipeline(graph_mode)

    evaluate_pipeline(
        exp_name=args.exp_name,
        model_pipeline=dofm_pipeline,
        model=model,
        args=args)
    
    print(f"--- Experiment Finished ---")
