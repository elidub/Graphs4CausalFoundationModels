"""
Test script to debug why predictions are constant.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, '/Users/arikreuter/Documents/PhD/CausalPriorFitting')

from src.models.PreprocessingGraphConditionedPFN import PreprocessingGraphConditionedPFN

# Model paths
checkpoint_path = "/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments/FirstTests/checkpoints/final_earlytest_16773250.0/final_model_with_bardist.pt"
model_config_path = "/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments/FirstTests/checkpoints/final_earlytest_16773250.0/final_model_with_bardist_config.yaml"

print(f"Loading model...")
model = PreprocessingGraphConditionedPFN(
    config_path=model_config_path,
    checkpoint_path=checkpoint_path,
    verbose=True,
)
model.load()

# Create synthetic data with clear structure
n_train = 200
n_test = 10
n_features = 5

np.random.seed(42)
X_train = np.random.randn(n_train, n_features).astype(np.float32)
t_train = np.random.binomial(1, 0.5, (n_train, 1)).astype(np.float32)
# True Y = sum(X) + 2*T + noise
y_train = (X_train.sum(axis=1, keepdims=True) + 2*t_train + np.random.randn(n_train, 1) * 0.1).astype(np.float32)

# Test data with varying features
X_test = np.random.randn(n_test, n_features).astype(np.float32)

print(f"\n=== Raw data statistics ===")
print(f"X_train shape: {X_train.shape}, range: [{X_train.min():.3f}, {X_train.max():.3f}]")
print(f"t_train shape: {t_train.shape}, mean: {t_train.mean():.3f}")
print(f"y_train shape: {y_train.shape}, range: [{y_train.min():.3f}, {y_train.max():.3f}]")
print(f"X_test shape: {X_test.shape}, range: [{X_test.min():.3f}, {X_test.max():.3f}]")

# Fit preprocessing
model.fit(X_train, t_train, y_train)

print(f"\n=== Preprocessing stats ===")
print(f"feature_means: {model._feature_means}")
print(f"feature_stds: {model._feature_stds}")
print(f"_y_min: {model._y_min}, _y_max: {model._y_max}")

# Check what the preprocessed data looks like
model_n_features = model.model.num_features
X_train_preprocessed = model._preprocess_features(X_train)
X_test_preprocessed = model._preprocess_features(X_test)

print(f"\n=== After preprocessing ===")
print(f"X_train_preprocessed shape: {X_train_preprocessed.shape}")
print(f"X_train_preprocessed range: [{X_train_preprocessed.min():.3f}, {X_train_preprocessed.max():.3f}]")
print(f"X_train_preprocessed first 5 features mean: {X_train_preprocessed[:, :5].mean():.3f}")
print(f"X_train_preprocessed padded features (5+) mean: {X_train_preprocessed[:, 5:].mean():.3f}")

# Build adjacency matrix (using [T, Y, X] format)
adj = np.zeros((model_n_features + 2, model_n_features + 2), dtype=np.float32)
T_idx = 0
Y_idx = 1
feature_offset = 2

adj[T_idx, Y_idx] = 1.0
for i in range(n_features):
    adj[feature_offset + i, T_idx] = 1.0
    adj[feature_offset + i, Y_idx] = 1.0
for i in range(n_features, model_n_features):
    feat_idx = feature_offset + i
    adj[feat_idx, :] = -1.0
    adj[:, feat_idx] = -1.0

# Make predictions
T_intv_1 = np.ones((n_test, 1), dtype=np.float32)
T_intv_0 = np.zeros((n_test, 1), dtype=np.float32)

print(f"\n=== Making predictions ===")
y_pred_1 = model.predict(X_obs=X_train, T_obs=t_train, Y_obs=y_train,
                          X_intv=X_test, T_intv=T_intv_1, 
                          adjacency_matrix=adj, prediction_type="mean")
y_pred_0 = model.predict(X_obs=X_train, T_obs=t_train, Y_obs=y_train,
                          X_intv=X_test, T_intv=T_intv_0, 
                          adjacency_matrix=adj, prediction_type="mean")

print(f"y_pred (T=1): {y_pred_1}")
print(f"y_pred (T=0): {y_pred_0}")
print(f"CATE: {y_pred_1 - y_pred_0}")
print(f"Mean CATE: {(y_pred_1 - y_pred_0).mean():.4f}")

# Check Y scaling
print(f"\n=== Checking Y target mapping ===")
# What value corresponds to mid-range of bar distribution?
print(f"BarDistribution centers range: {model.model.bar_distribution.centers[0].item():.3f} to {model.model.bar_distribution.centers[-1].item():.3f}")
print(f"Target was scaled to [-1, 1] range from [{model._y_min:.3f}, {model._y_max:.3f}]")
print(f"Predictions (before inverse transform): checking if they're in [-1, 1] range")

# Manually check raw model output
print(f"\n=== Raw model output (no inverse transform) ===")
y_pred_1_raw = model.predict(X_obs=X_train, T_obs=t_train, Y_obs=y_train,
                              X_intv=X_test, T_intv=T_intv_1, 
                              adjacency_matrix=adj, prediction_type="mean",
                              inverse_transform=False)
y_pred_0_raw = model.predict(X_obs=X_train, T_obs=t_train, Y_obs=y_train,
                              X_intv=X_test, T_intv=T_intv_0, 
                              adjacency_matrix=adj, prediction_type="mean",
                              inverse_transform=False)
print(f"y_pred_raw (T=1): {y_pred_1_raw}")
print(f"y_pred_raw (T=0): {y_pred_0_raw}")
print(f"Raw CATE: {y_pred_1_raw - y_pred_0_raw}")

# Check bar distribution range
print(f"\n=== Bar distribution info ===")
bd = model.model.bar_distribution
print(f"Centers shape: {bd.centers.shape}")
print(f"Centers range: [{bd.centers.min():.4f}, {bd.centers.max():.4f}]")
print(f"Edges shape: {bd.edges.shape}")
print(f"Edges range: [{bd.edges.min():.4f}, {bd.edges.max():.4f}]")
