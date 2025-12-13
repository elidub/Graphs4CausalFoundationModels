"""
Simplified script for evaluating Observational and Interventional PFN models.
This script provides reusable functions for running experiments across different SCM configurations.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src" / "models"))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Import custom modules
from src.models.InterventionalPFN_sklearn import InterventionalPFNSklearn
from src.models.SimplePFN_sklearn import SimplePFNSklearn
from src.priordata_processing.BasicProcessing import BasicProcessing
from src.priors.causal_prior.causal_graph.GraphSampler import GraphSampler
from src.priors.causal_prior.causal_graph.CausalDAG import CausalDAG
from src.priors.causal_prior.scm.SCM import SCM
from src.priors.causal_prior.noise_distributions.NormalDistribution import NormalDistribution
from src.priors.causal_prior.noise_distributions.ResamplingDist import ResamplingDist
from src.priors.causal_prior.mechanisms.LinearMechanism import LinearMechanism


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def visualize_dataset(data_dict, title="Dataset Visualization", figsize=None):
    """
    Visualize a dataset with pairwise scatterplots and histograms on the diagonal.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary where keys are variable names and values are tensors
    title : str
        Title for the plot
    figsize : tuple, optional
        Figure size. If None, automatically determined
    """
    # Convert tensors to numpy and create DataFrame
    df_data = {}
    for key, value in data_dict.items():
        if hasattr(value, 'numpy'):
            df_data[f'X{key}'] = value.numpy().flatten()
        else:
            df_data[f'X{key}'] = value.flatten()
    
    df = pd.DataFrame(df_data)
    n_vars = len(df.columns)
    
    if figsize is None:
        figsize = (3 * n_vars, 3 * n_vars)
    
    fig, axes = plt.subplots(n_vars, n_vars, figsize=figsize)
    
    if n_vars == 1:
        axes = np.array([[axes]])
    
    for i in range(n_vars):
        for j in range(n_vars):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: histogram
                ax.hist(df.iloc[:, i], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
                ax.set_ylabel('Frequency')
            else:
                # Off-diagonal: scatterplot
                ax.scatter(df.iloc[:, j], df.iloc[:, i], alpha=0.5, s=10)
            
            # Labels
            if i == n_vars - 1:
                ax.set_xlabel(df.columns[j])
            else:
                ax.set_xticklabels([])
            
            if j == 0 and i != j:
                ax.set_ylabel(df.columns[i])
            else:
                ax.set_yticklabels([])
            
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, y=0.995)
    plt.tight_layout()
    return fig


def load_models(obspfn_config, obspfn_checkpoint, intpfn_config, intpfn_checkpoint, verbose=True):
    """
    Load the observational and interventional PFN models.
    
    Parameters:
    -----------
    obspfn_config : str
        Path to observational PFN config
    obspfn_checkpoint : str
        Path to observational PFN checkpoint
    intpfn_config : str
        Path to interventional PFN config
    intpfn_checkpoint : str
        Path to interventional PFN checkpoint
    verbose : bool
        Enable verbose output
        
    Returns:
    --------
    obspfn, intpfn : SimplePFNSklearn, InterventionalPFNSklearn
        Loaded models
    """
    print("Loading Observational PFN...")
    obspfn = SimplePFNSklearn(
        config_path=obspfn_config,
        checkpoint_path=obspfn_checkpoint,
        n_estimators=1,
        verbose=verbose,
        max_n_train=None,
    )
    obspfn.load()
    
    print("Loading Interventional PFN...")
    intpfn = InterventionalPFNSklearn(
        config_path=intpfn_config,
        checkpoint_path=intpfn_checkpoint,
        n_estimators=1,
        verbose=verbose,
        max_n_train=None,
    )
    intpfn.load()
    
    return obspfn, intpfn


def create_processor(n_features=49, target_feature=2, intervened_feature=0, 
                     n_samples=500, feature_standardize=True):
    """
    Create a BasicProcessing instance with specified parameters.
    
    Parameters:
    -----------
    n_features : int
        Number of features (excluding target)
    target_feature : int
        Index of target feature
    intervened_feature : int
        Index of intervention feature
    n_samples : int
        Number of training/test samples
    feature_standardize : bool
        Whether to standardize features
        
    Returns:
    --------
    processor : BasicProcessing
        Configured processor
    """
    return BasicProcessing(
        n_features=n_features,
        max_n_features=n_features,
        n_train_samples=n_samples,
        max_n_train_samples=n_samples,
        n_test_samples=n_samples,
        max_n_test_samples=n_samples,
        dropout_prob=0.0,
        target_feature=target_feature,
        intervened_feature=intervened_feature,
        feature_standardize=feature_standardize,
        shuffle_features=False,
        shuffle_samples=False,
        feature_negative_one_one_scaling=False,
        target_negative_one_one_scaling=True,
        remove_outliers=True,
        outlier_quantile=0.99,
        yeo_johnson=False
    )


def generate_scm_data(graph_edges, mechanisms, intervention_node=0, 
                     num_train_samples=500, num_test_samples=500, seed=42):
    """
    Generate observational and interventional data from an SCM.
    
    Parameters:
    -----------
    graph_edges : list of tuples
        List of directed edges (from, to)
    mechanisms : dict
        Dictionary mapping node indices to mechanism functions
    intervention_node : int
        Node to intervene on
    num_train_samples : int
        Number of training samples
    num_test_samples : int
        Number of test samples
    seed : int
        Random seed
        
    Returns:
    --------
    obs0, obs1, interv1 : dict, dict, dict
        Training observational, test observational, and interventional data
    """
    num_nodes = max(max(edge) for edge in graph_edges) + 1
    
    # Create graph
    graph_sampler = GraphSampler(seed=seed)
    graph = graph_sampler.sample_dag(num_nodes=num_nodes, p=0.5)
    graph.clear_edges()
    graph.add_edges_from(graph_edges)
    
    causal_dag = CausalDAG(g=graph, check_acyclic=True)
    
    # Setup noise distributions
    exogenous_variables = causal_dag.exogenous_variables()
    endogenous_variables = causal_dag.endogenous_variables()
    
    exo_noise = {var: NormalDistribution(mean=0, std=1) for var in exogenous_variables}
    endo_noise = {var: NormalDistribution(mean=0, std=1) for var in endogenous_variables}
    
    # Create SCM
    scm = SCM(
        dag=causal_dag,
        mechanisms=mechanisms,
        exogenous_noise=exo_noise,
        endogenous_noise=endo_noise,
        use_exogenous_mechanisms=False
    )
    
    # Sample training observational data
    scm.sample_exogenous(num_samples=num_train_samples)
    scm.sample_endogenous(num_samples=num_train_samples)
    obs0_raw = scm.propagate(num_samples=num_train_samples)
    obs0 = {k: v.reshape(-1, 1) if v.dim() == 1 else v for k, v in obs0_raw.items()}
    
    # Sample test observational data
    scm.sample_exogenous(num_samples=num_test_samples)
    scm.sample_endogenous(num_samples=num_test_samples)
    obs1_raw = scm.propagate(num_samples=num_test_samples)
    obs1 = {k: v.reshape(-1, 1) if v.dim() == 1 else v for k, v in obs1_raw.items()}
    
    # Perform intervention
    intervention_samples = obs1_raw[intervention_node]
    interventional_dist = ResamplingDist(intervention_samples)
    
    scm.intervene(node=intervention_node)
    
    if intervention_node in scm.dag.endogenous_variables():
        scm.endogenous_noise[intervention_node] = interventional_dist
    if intervention_node in scm.dag.exogenous_variables():
        scm.exogenous_noise[intervention_node] = interventional_dist
    
    # Sample interventional data
    scm.sample_exogenous(num_samples=num_test_samples)
    scm.sample_endogenous(num_samples=num_test_samples)
    interv1_raw = scm.propagate(num_samples=num_test_samples)
    interv1 = {k: v.reshape(-1, 1) if v.dim() == 1 else v for k, v in interv1_raw.items()}
    
    return obs0, obs1, interv1, causal_dag


def evaluate_models(obspfn, intpfn, processor, obs0, obs1, interv1):
    """
    Evaluate all three models (RF, ObsPFN, IntPFN) on train, obs2, and test sets.
    
    Parameters:
    -----------
    obspfn : SimplePFNSklearn
        Observational PFN model
    intpfn : InterventionalPFNSklearn
        Interventional PFN model
    processor : BasicProcessing
        Data processor
    obs0, obs1, interv1 : dict
        Training obs, test obs, and interventional data
        
    Returns:
    --------
    results : dict
        Dictionary containing all metrics for all models and datasets
    """
    # Process data
    X_train, T_train, Y_train, X_test, T_test, Y_test = processor.process_from_splits(obs0, interv1)
    X_obs2, T_obs2, Y_obs2, _, _, _ = processor.process_from_splits(obs1, interv1)
    
    XT_train = torch.cat([X_train, T_train], dim=1)
    XT_test = torch.cat([X_test, T_test], dim=1)
    XT_obs2 = torch.cat([X_obs2, T_obs2], dim=1)
    
    results = {}
    
    # ========== RANDOM FOREST ==========
    rf = RandomForestRegressor(n_estimators=10)
    rf.fit(XT_train.numpy(), Y_train.numpy().ravel())
    
    for dataset_name, XT, Y in [('train', XT_train, Y_train), 
                                  ('obs2', XT_obs2, Y_obs2), 
                                  ('test', XT_test, Y_test)]:
        preds = rf.predict(XT.numpy())
        y_true = Y.numpy().ravel()
        squared_errors = (y_true - preds)**2
        
        results[f'rf_{dataset_name}_mse'] = np.mean(squared_errors)
        results[f'rf_{dataset_name}_mse_std'] = np.std(squared_errors) / np.sqrt(len(squared_errors))
        results[f'rf_{dataset_name}_bias'] = np.mean(preds) - np.mean(y_true)
        results[f'rf_{dataset_name}_variance'] = np.var(preds)
        
        # R² score
        ss_res = np.sum(squared_errors)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        results[f'rf_{dataset_name}_r2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # ========== OBSERVATIONAL PFN ==========
    for dataset_name, XT, Y in [('train', XT_train, Y_train), 
                                  ('obs2', XT_obs2, Y_obs2), 
                                  ('test', XT_test, Y_test)]:
        preds = obspfn.predict(
            X_train=XT_train.numpy(),
            y_train=Y_train.squeeze().numpy(),
            X_test=XT.numpy(),
            prediction_type="mean"
        )
        y_true = Y.numpy().ravel()
        squared_errors = (y_true - preds)**2
        
        results[f'obspfn_{dataset_name}_mse'] = np.mean(squared_errors)
        results[f'obspfn_{dataset_name}_mse_std'] = np.std(squared_errors) / np.sqrt(len(squared_errors))
        results[f'obspfn_{dataset_name}_bias'] = np.mean(preds) - np.mean(y_true)
        results[f'obspfn_{dataset_name}_variance'] = np.var(preds)
        
        # R² score
        ss_res = np.sum(squared_errors)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        results[f'obspfn_{dataset_name}_r2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        log_lik = obspfn.predict_log_likelihood(
            X_train=XT_train.numpy(),
            y_train=Y_train.squeeze().numpy(),
            X_test=XT.numpy(),
            y_test=Y.numpy()
        )
        results[f'obspfn_{dataset_name}_nll'] = -np.mean(log_lik)
        results[f'obspfn_{dataset_name}_nll_std'] = np.std(-log_lik) / np.sqrt(len(log_lik))
    
    # ========== INTERVENTIONAL PFN ==========
    X_obs_ext = torch.cat([X_train, torch.zeros_like(T_train)], dim=1)
    X_intv_ext = torch.cat([X_test, torch.zeros_like(T_test)], dim=1)
    X_obs2_ext = torch.cat([X_obs2, torch.zeros_like(T_obs2)], dim=1)
    
    datasets = [
        ('train', X_obs_ext, T_train, Y_train),
        ('obs2', X_obs2_ext, T_obs2, Y_obs2),
        ('test', X_intv_ext, T_test, Y_test)
    ]
    
    for dataset_name, X_intv, T_intv, Y in datasets:
        preds = intpfn.predict(
            X_obs=X_obs_ext.numpy(),
            T_obs=T_train.numpy(),
            Y_obs=Y_train.squeeze().numpy(),
            X_intv=X_intv.numpy(),
            T_intv=T_intv.numpy(),
            prediction_type="mean"
        )
        y_true = Y.numpy().ravel()
        squared_errors = (y_true - preds)**2
        
        results[f'intpfn_{dataset_name}_mse'] = np.mean(squared_errors)
        results[f'intpfn_{dataset_name}_mse_std'] = np.std(squared_errors) / np.sqrt(len(squared_errors))
        results[f'intpfn_{dataset_name}_bias'] = np.mean(preds) - np.mean(y_true)
        results[f'intpfn_{dataset_name}_variance'] = np.var(preds)
        
        # R² score
        ss_res = np.sum(squared_errors)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        results[f'intpfn_{dataset_name}_r2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        log_lik = intpfn.predict_log_likelihood(
            X_obs=X_obs_ext.numpy(),
            T_obs=T_train.numpy(),
            Y_obs=Y_train.squeeze().numpy(),
            X_intv=X_intv.numpy(),
            T_intv=T_intv.numpy(),
            Y_intv=Y.numpy()
        )
        results[f'intpfn_{dataset_name}_nll'] = -np.mean(log_lik)
        results[f'intpfn_{dataset_name}_nll_std'] = np.std(-log_lik) / np.sqrt(len(log_lik))
    
    return results


def print_results(results, case_name=""):
    """
    Print formatted results for all models and datasets.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing all metrics
    case_name : str
        Name of the case study
    """
    if case_name:
        print(f"\n{'='*140}")
        print(f"CASE STUDY: {case_name}")
        print(f"{'='*140}\n")
    
    datasets = ['train', 'obs2', 'test']
    dataset_labels = {
        'train': 'TRAIN SET (obs0 - used for training)',
        'obs2': 'OBS2 SET (obs1 - second observational, not trained on)',
        'test': 'TEST SET (interventional)'
    }
    
    for dataset in datasets:
        print("="*140)
        print(f"SUMMARY - {dataset_labels[dataset]}")
        print("="*140)
        print(f"{'Method':<20} {'MSE (±SE)':<25} {'R²':<12} {'Bias':<15} {'Variance':<15} {'NLL (±SE)':<25}")
        print("-"*140)
        
        # Random Forest
        mse = results[f'rf_{dataset}_mse']
        mse_std = results[f'rf_{dataset}_mse_std']
        r2 = results[f'rf_{dataset}_r2']
        bias = results[f'rf_{dataset}_bias']
        var = results[f'rf_{dataset}_variance']
        print(f"{'Random Forest':<20} {mse:.6f} (±{mse_std:.6f}){'':<3} {r2:<12.6f} {bias:<15.6f} {var:<15.6f} {'N/A':<25}")
        
        # Observational PFN
        mse = results[f'obspfn_{dataset}_mse']
        mse_std = results[f'obspfn_{dataset}_mse_std']
        r2 = results[f'obspfn_{dataset}_r2']
        bias = results[f'obspfn_{dataset}_bias']
        var = results[f'obspfn_{dataset}_variance']
        nll = results[f'obspfn_{dataset}_nll']
        nll_std = results[f'obspfn_{dataset}_nll_std']
        print(f"{'Obs PFN':<20} {mse:.6f} (±{mse_std:.6f}){'':<3} {r2:<12.6f} {bias:<15.6f} {var:<15.6f} {nll:.6f} (±{nll_std:.6f})")
        
        # Interventional PFN
        mse = results[f'intpfn_{dataset}_mse']
        mse_std = results[f'intpfn_{dataset}_mse_std']
        r2 = results[f'intpfn_{dataset}_r2']
        bias = results[f'intpfn_{dataset}_bias']
        var = results[f'intpfn_{dataset}_variance']
        nll = results[f'intpfn_{dataset}_nll']
        nll_std = results[f'intpfn_{dataset}_nll_std']
        print(f"{'Int PFN':<20} {mse:.6f} (±{mse_std:.6f}){'':<3} {r2:<12.6f} {bias:<15.6f} {var:<15.6f} {nll:.6f} (±{nll_std:.6f})")
        print("="*140)
        print()


# ============================================================================
# CASE STUDY DEFINITIONS
# ============================================================================

def case_study_1_simple_chain():
    """
    Three variables: X0 -> X2, X1 -> X2
    No hidden confounders, simple linear mechanisms.
    """
    graph_edges = [(1, 2), (0, 2)]
    
    mechanisms = {
        2: LinearMechanism(
            input_dim=2,
            weights=[1.0, 1.0],
            nonlinearity=lambda x: torch.tanh(x)
        )
    }
    
    return graph_edges, mechanisms, 0  # intervention_node=0


def case_study_2_with_mediator():
    """
    Three variables: X0 -> X1 -> X2, X0 -> X2
    Includes a mediator variable.
    """
    graph_edges = [(1, 2), (0, 1), (0, 2)]
    
    mechanisms = {
        2: LinearMechanism(
            input_dim=2,
            weights=[1.0, 1.0],
            nonlinearity=lambda x: torch.tanh(x)
        ),
        1: LinearMechanism(
            input_dim=1,
            weights=[1.0],
            nonlinearity=lambda x: torch.tanh(x)
        )
    }
    
    return graph_edges, mechanisms, 0  # intervention_node=0


def case_study_3_four_variables():
    """
    Four variables with more complex dependencies.
    """
    graph_edges = [
            (1, 2), 
            (0, 1), 
            (3, 0),
            (3,2),
            #(0,2)
            ]
    
    mechanisms = {
        0: LinearMechanism(
            input_dim=1,
            weights=[10.0],   # was 3.0 : stronger U -> T
            nonlinearity=lambda x: torch.tanh(x),
        ),
        1: LinearMechanism(
            input_dim = 1,
            weights = [1.0],
            nonlinearity= lambda x: torch.tanh(x)
        ),
        2: LinearMechanism(
            input_dim=2,
            weights=[5.0, 10.0],  # make U -> Y big
            nonlinearity=lambda x: torch.tanh(x),       
        ),

        3: LinearMechanism(
            input_dim=2,
            weights=[1.0, 1.0],
            nonlinearity=lambda x: torch.tanh(x)
        )
    }
    
    return graph_edges, mechanisms, 0  # intervention_node=0


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function - runs all case studies.
    """
    # Configuration paths
    OBSPFN_CONFIG = "/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments/FirstTests/configs/basic_predictive.yaml"
    OBSPFN_CHECKPOINT = "/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments/FirstTests/checkpoints/simple_pfn_16661179.0/step_49000.pt"
    INTPFN_CONFIG = "/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments/FirstTests/configs/basic.yaml"
    INTPFN_CHECKPOINT = "/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments/FirstTests/checkpoints/simple_pfn_16661592.0/step_30000_92.pt"
    
    # Load models once
    obspfn, intpfn = load_models(OBSPFN_CONFIG, OBSPFN_CHECKPOINT, 
                                   INTPFN_CONFIG, INTPFN_CHECKPOINT, verbose=True)
    
    # Create processor
    processor = create_processor(n_features=49, target_feature=2, 
                                 intervened_feature=0, n_samples=500)
    
    # Define case studies
    case_studies = [
        ("Simple Chain (X0->X2, X1->X2)", case_study_1_simple_chain),
        ("With Mediator (X0->X1->X2, X0->X2)", case_study_2_with_mediator),
        ("Four Variables", case_study_3_four_variables),
    ]
    
    # Run each case study
    for case_name, case_func in case_studies:
        print(f"\n\n{'#'*120}")
        print(f"# Running: {case_name}")
        print(f"{'#'*120}\n")
        
        # Get case-specific configuration
        graph_edges, mechanisms, intervention_node = case_func()

        
        
        # Generate data
        obs0, obs1, interv1, causal_dag = generate_scm_data(
            graph_edges=graph_edges,
            mechanisms=mechanisms,
            intervention_node=intervention_node,
            num_train_samples=500,
            num_test_samples=500
        )

        causal_dag.draw()
        plt.title(f"Causal Graph: {case_name}")
        plt.show()
        
        # Visualize graph
       
        
        # Evaluate models
        results = evaluate_models(obspfn, intpfn, processor, obs0, obs1, interv1)
        
        # Print results
        print_results(results, case_name)


if __name__ == "__main__":
    main()
