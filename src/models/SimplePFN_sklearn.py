from __future__ import annotations
from typing import Optional, Dict, Any, Union, Literal

import yaml
import torch
import numpy as np
import re
from pathlib import Path

# Robust import: try package-style, then relative, then add repo root to sys.path
try:
    from src.models.SimplePFN import SimplePFNRegressor
    from src.Losses.BarDistribution import BarDistribution
except Exception:
    try:
        from models.SimplePFN import SimplePFNRegressor
        from Losses.BarDistribution import BarDistribution
    except Exception:
        import sys
        repo_root = Path(__file__).resolve().parents[2]
        sys.path.append(str(repo_root))
        from src.models.SimplePFN import SimplePFNRegressor
        from src.Losses.BarDistribution import BarDistribution


class SimplePFNSklearn:
    """
    A small scikit-learn-like wrapper around the SimplePFN PyTorch model with BarDistribution support.

    Usage:
      wrapper = SimplePFNSklearn(config_path="path/to/config.yaml", checkpoint_path="model.pt")
      wrapper.load()                  # builds model and loads weights, sets up BarDistribution if configured
      
      # For point predictions (mode/mean of posterior)
      preds = wrapper.predict(X_train, y_train, X_test, prediction_type="mode")  # or "mean"
      
      # For probabilistic predictions (samples from posterior)
      samples = wrapper.predict(X_train, y_train, X_test, prediction_type="sample", num_samples=100)

    Notes:
    - `fit` is a no-op placeholder (training is handled elsewhere); it returns self to
      satisfy the sklearn convention.
    - `predict` accepts numpy arrays or torch tensors. It accepts both batched inputs
      (B, N, F) and single-dataset inputs (N, F) which are treated as batch size 1.
    - When BarDistribution is enabled (use_bar_distribution=true in config), the model
      outputs high-dimensional parameters that are processed through BarDistribution
      to extract posterior statistics.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = "cpu",
        verbose: bool = False,
    ):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.verbose = verbose

        self.model: Optional[SimplePFNRegressor] = None
        self.model_kwargs: Dict[str, Any] = {}
        self.bar_distribution: Optional[BarDistribution] = None
        self.use_bar_distribution: bool = False
    def load(self, override_kwargs: Optional[Dict[str, Any]] = None) -> "SimplePFNSklearn":
        """Load config (if provided), apply optional override_kwargs, build the model and load checkpoint (if provided).

        override_kwargs: a dict of model kwargs (e.g., {'num_features': 9}) that will update values from the YAML.
        """
        if self.config_path:
            with open(self.config_path, "r") as f:
                cfg = yaml.safe_load(f)
            # model config lives under top-level 'model_config'
            mcfg = cfg.get("model_config", {}) if isinstance(cfg, dict) else {}
            
            # Check for BarDistribution configuration
            self.use_bar_distribution = mcfg.get("use_bar_distribution", {}).get("value", False)
            
            # map expected keys
            self.model_kwargs = {
                "num_features": int(mcfg.get("num_features", {}).get("value", 0)),
                "d_model": int(mcfg.get("d_model", {}).get("value", 256)),
                "depth": int(mcfg.get("depth", {}).get("value", 8)),
                "heads_feat": int(mcfg.get("heads_feat", {}).get("value", 8)),
                "heads_samp": int(mcfg.get("heads_samp", {}).get("value", 8)),
                "dropout": float(mcfg.get("dropout", {}).get("value", 0.0)),
                "hidden_mult": int(mcfg.get("hidden_mult", {}).get("value", 4)),
            }
            
            # Set output_dim based on BarDistribution configuration
            if self.use_bar_distribution:
                num_bars = int(mcfg.get("num_bars", {}).get("value", 11))
                output_dim = num_bars + 4  # BarDistribution requires K + 4 parameters
                self.model_kwargs["output_dim"] = output_dim
                
                # Create BarDistribution instance
                self.bar_distribution = BarDistribution(
                    num_bars=num_bars,
                    min_width=float(mcfg.get("min_width", {}).get("value", 1e-6)),
                    scale_floor=float(mcfg.get("scale_floor", {}).get("value", 1e-6)),
                    device=self.device,
                    max_fit_items=mcfg.get("max_fit_items", {}).get("value", None),
                    log_prob_clip_min=float(mcfg.get("log_prob_clip_min", {}).get("value", -50.0)),
                    log_prob_clip_max=float(mcfg.get("log_prob_clip_max", {}).get("value", 50.0)),
                )
                
                if self.verbose:
                    print(f"[SimplePFNSklearn] BarDistribution enabled with {num_bars} bars, output_dim={output_dim}")
            else:
                self.model_kwargs["output_dim"] = 1
                if self.verbose:
                    print("[SimplePFNSklearn] Using standard MSE output (output_dim=1)")
        else:
            # If the user provided model_kwargs before load(), that's acceptable.
            if self.model_kwargs and self.model_kwargs.get("num_features"):
                if self.verbose:
                    print("[SimplePFNSklearn] Using pre-specified model_kwargs.")
            else:
                if self.verbose:
                    print("[SimplePFNSklearn] No config_path provided; you must supply model kwargs manually before load().")

        # apply overrides if present (these take precedence over config)
        if override_kwargs:
            self.model_kwargs.update({k: v for k, v in override_kwargs.items() if v is not None})

        # sanity check for num_features
        if not self.model_kwargs.get("num_features"):
            # If num_features is missing, build a minimal placeholder (user must ensure correctness)
            self.model_kwargs.setdefault("num_features", 1)

        # build model
        self.model = SimplePFNRegressor(**self.model_kwargs).to(self.device)

        # load weights
        if self.checkpoint_path:
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            # common wrappers store state in ['state_dict'] or ['model_state_dict']
            state = None
            if isinstance(ckpt, dict):
                for key in ("state_dict", "model_state_dict", "net", "model"):
                    if key in ckpt:
                        state = ckpt[key]
                        break
                if state is None:
                    # maybe the dict is already the state_dict
                    state = ckpt
            else:
                state = ckpt

            # some state_dicts have 'module.' prefixes from DataParallel; strip if needed
            def _strip_module(sdict):
                new = {}
                for k, v in sdict.items():
                    nk = k.replace("module.", "")
                    new[nk] = v
                return new

            if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
                state = _strip_module(state)

            # Try loading; if size mismatch occurs, attempt to infer d_model/depth from checkpoint and rebuild
            try:
                self.model.load_state_dict(state, strict=False)
                if self.verbose:
                    print("[SimplePFNSklearn] Loaded checkpoint into model (partial loads allowed).")
            except Exception as e:
                if self.verbose:
                    print(f"[SimplePFNSklearn] Warning: initial load failed: {e}")
                # attempt to infer d_model from common param shapes
                try:
                    inferred = {}
                    # label_mask_embed -> shape (1,1,d)
                    if "label_mask_embed" in state:
                        v = state["label_mask_embed"]
                        if v.ndim >= 3:
                            inferred["d_model"] = int(v.shape[-1])
                    # value_encoder.weight -> shape (d_model, 1)
                    if "value_encoder.weight" in state and "d_model" not in inferred:
                        v = state["value_encoder.weight"]
                        inferred["d_model"] = int(v.shape[0])
                    # count blocks to infer depth
                    block_idxs = [int(m.group(1)) for k in state.keys() for m in [re.match(r"blocks\.(\d+)\.", k)] if m]
                    if block_idxs:
                        inferred["depth"] = int(max(block_idxs) + 1)

                    if inferred:
                        if self.verbose:
                            print(f"[SimplePFNSklearn] Inferred from checkpoint: {inferred}")
                        # update kwargs and rebuild model
                        if "d_model" in inferred:
                            self.model_kwargs["d_model"] = inferred["d_model"]
                        if "depth" in inferred:
                            self.model_kwargs["depth"] = inferred["depth"]
                        # rebuild model with inferred d_model/depth
                        self.model = SimplePFNRegressor(**self.model_kwargs).to(self.device)
                        try:
                            self.model.load_state_dict(state, strict=False)
                            if self.verbose:
                                print("[SimplePFNSklearn] Successfully rebuilt model with inferred shape and loaded checkpoint.")
                        except Exception as e2:
                            if self.verbose:
                                print(f"[SimplePFNSklearn] Warning: failed to load even after inferring shapes: {e2}")
                    else:
                        if self.verbose:
                            print("[SimplePFNSklearn] Could not infer model shapes from checkpoint; skipping weight load.")
                except Exception as e3:
                    if self.verbose:
                        print(f"[SimplePFNSklearn] Error while attempting to infer shapes: {e3}")

        return self

    def fit_bar_distribution(self, X_train_data, y_train_data, X_test_data, y_test_data, 
                           max_batches: Optional[int] = None) -> "SimplePFNSklearn":
        """
        Fit the BarDistribution to data. This must be called before prediction if BarDistribution is enabled.
        
        Args:
            X_train_data, y_train_data, X_test_data, y_test_data: Training and test data arrays
                Can be lists of arrays (multiple datasets) or single arrays (single dataset)
            max_batches: Maximum number of batches to use for fitting
        """
        if not self.use_bar_distribution or self.bar_distribution is None:
            if self.verbose:
                print("[SimplePFNSklearn] BarDistribution not enabled, skipping fit.")
            return self
            
        # Convert data to proper format for BarDistribution fitting
        if not isinstance(X_train_data, list):
            X_train_data = [X_train_data]
            y_train_data = [y_train_data] 
            X_test_data = [X_test_data]
            y_test_data = [y_test_data]
            
        # Create a simple iterator that yields (X_train, y_train, X_test, y_test) tuples
        class SimpleDataIterator:
            def __init__(self, X_tr_list, y_tr_list, X_te_list, y_te_list):
                self.data = list(zip(X_tr_list, y_tr_list, X_te_list, y_te_list))
                
            def __iter__(self):
                for X_tr, y_tr, X_te, y_te in self.data:
                    # Convert to tensors and ensure proper shapes
                    X_tr_tensor = torch.as_tensor(np.asarray(X_tr), dtype=torch.float32)
                    y_tr_tensor = torch.as_tensor(np.asarray(y_tr), dtype=torch.float32)
                    X_te_tensor = torch.as_tensor(np.asarray(X_te), dtype=torch.float32)
                    y_te_tensor = torch.as_tensor(np.asarray(y_te), dtype=torch.float32)
                    
                    # Ensure batch dimensions
                    if X_tr_tensor.ndim == 2:
                        X_tr_tensor = X_tr_tensor.unsqueeze(0)
                    if X_te_tensor.ndim == 2:
                        X_te_tensor = X_te_tensor.unsqueeze(0)
                    if y_tr_tensor.ndim == 1:
                        y_tr_tensor = y_tr_tensor.unsqueeze(0)
                    if y_te_tensor.ndim == 1:
                        y_te_tensor = y_te_tensor.unsqueeze(0)
                        
                    yield (X_tr_tensor, y_tr_tensor, X_te_tensor, y_te_tensor)
        
        data_iter = SimpleDataIterator(X_train_data, y_train_data, X_test_data, y_test_data)
        
        if self.verbose:
            print(f"[SimplePFNSklearn] Fitting BarDistribution with {len(X_train_data)} datasets...")
            
        self.bar_distribution.fit(data_iter, max_batches=max_batches)
        
        if self.verbose:
            print("[SimplePFNSklearn] BarDistribution fitted successfully.")
            
        return self

    def fit(self, X: Any = None, y: Any = None, **kwargs) -> "SimplePFNSklearn":
        """Placeholder fit to satisfy sklearn interface. Training should be done with the project's trainer.

        If checkpoint was provided `load()` should have been called before `fit`.
        """
        # no-op: we don't implement training here
        if self.verbose:
            print("[SimplePFNSklearn] fit() is a no-op placeholder. Use the training scripts to train the model.")
        return self

    def predict(self, X_train: Any, y_train: Any, X_test: Any, 
                prediction_type: Literal["point", "mode", "mean", "sample"] = "point",
                num_samples: int = 100) -> np.ndarray:
        """
        Run the PFN forward pass and return predictions as a numpy array.

        Args:
            X_train, y_train, X_test: Input data (numpy arrays or torch tensors)
                Shapes accepted:
                - X_train: (N, F) or (B, N, F)
                - y_train: (N,) or (N,1) or (B, N) or (B,N,1)  
                - X_test: (M, F) or (B, M, F)
            prediction_type: Type of prediction to return
                - "point": Raw model output (default for MSE) or mode (default for BarDistribution)
                - "mode": Posterior mode (requires BarDistribution)
                - "mean": Posterior mean (requires BarDistribution)
                - "sample": Samples from posterior (requires BarDistribution)
            num_samples: Number of samples to return when prediction_type="sample"

        Returns:
            numpy array of predictions:
            - For "point", "mode", "mean": shape (B, M) or (M,) for batch size 1
            - For "sample": shape (B, num_samples, M) or (num_samples, M) for batch size 1
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call load() before predict().")

        # Validate prediction type
        if prediction_type in ["mode", "mean", "sample"] and not self.use_bar_distribution:
            raise ValueError(f"prediction_type='{prediction_type}' requires BarDistribution to be enabled in config.")
        
        if self.use_bar_distribution and self.bar_distribution is None:
            raise RuntimeError("BarDistribution enabled but not fitted. Call fit_bar_distribution() first.")

        # convert to numpy if pandas
        if hasattr(X_train, "values"):
            X_train = X_train.values
        if hasattr(X_test, "values"):
            X_test = X_test.values
        if hasattr(y_train, "values"):
            y_train = y_train.values

        Xtr = torch.as_tensor(np.asarray(X_train), dtype=torch.float32)
        Xte = torch.as_tensor(np.asarray(X_test), dtype=torch.float32)
        ytr = torch.as_tensor(np.asarray(y_train), dtype=torch.float32)

        # normalize dims to (B, N, F) and (B, N) for y
        if Xtr.ndim == 2:
            Xtr = Xtr.unsqueeze(0)
        if Xte.ndim == 2:
            Xte = Xte.unsqueeze(0)

        if ytr.ndim == 1:
            ytr = ytr.unsqueeze(0)

        # ensure y has shape (B, N) or (B, N, 1) acceptable to model
        if ytr.ndim == 3 and ytr.shape[-1] == 1:
            pass
        elif ytr.ndim == 2:
            # ok
            pass
        else:
            # try to squeeze last dim
            ytr = ytr.squeeze(-1)

        # move to device
        Xtr = Xtr.to(self.device)
        Xte = Xte.to(self.device)
        ytr = ytr.to(self.device)

        # shape checks
        B, N, F = Xtr.shape
        _, M, F2 = Xte.shape
        if F != self.model.num_features:
            raise ValueError(f"Model expects num_features={self.model.num_features}, but got input with {F} features")

        # run model
        self.model.eval()
        with torch.no_grad():
            out = self.model(Xtr, ytr, Xte)
            raw_predictions = out["predictions"]  # Shape depends on output_dim

        # Process predictions based on type and BarDistribution usage
        if self.use_bar_distribution:
            # Model outputs BarDistribution parameters
            if prediction_type in ["point", "mode"]:
                result = self.bar_distribution.mode(raw_predictions).cpu().numpy()
            elif prediction_type == "mean":
                result = self.bar_distribution.mean(raw_predictions).cpu().numpy()
            elif prediction_type == "sample":
                result = self.bar_distribution.sample(raw_predictions, num_samples).cpu().numpy()
                # BarDistribution returns (B, num_samples, M), which is what we want
            else:
                raise ValueError(f"Unknown prediction_type: {prediction_type}")
        else:
            # Standard MSE model output
            if prediction_type != "point":
                raise ValueError(f"prediction_type='{prediction_type}' not supported without BarDistribution")
            result = raw_predictions.cpu().numpy()
            
        # Handle single batch case for consistency
        if result.shape[0] == 1 and prediction_type != "sample":
            return result[0]
        elif result.shape[0] == 1 and prediction_type == "sample":
            return result[0]  # Return (num_samples, M) for single batch
        
        return result


if __name__ == "__main__":
    # Run a small end-to-end inference test using the project config and checkpoint
    import sys
    sys.path.append("/fast/arikreuter/DoPFN_v2/CausalPriorFitting")

    cfg_path = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/configs/early_test.yaml"
    ckpt_path = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/checkpoints/early_test1_32bs/step_100000.pt"

    print("[SimplePFNSklearn] Building wrapper with config and checkpoint...")
    w = SimplePFNSklearn(config_path=cfg_path, checkpoint_path=ckpt_path, device="cpu", verbose=True)
    w.load()
    print("[SimplePFNSklearn] Model built. model_kwargs:", w.model_kwargs)
    print(f"[SimplePFNSklearn] BarDistribution enabled: {w.use_bar_distribution}")

    # prepare a tiny synthetic batch: 10 train samples, 3 test samples
    num_features = int(w.model.num_features)
    print(f"[SimplePFNSklearn] num_features = {num_features}")

    import numpy as _np
    Xtr = _np.random.randn(10, num_features).astype(_np.float32)
    ytr = _np.random.randn(10).astype(_np.float32)
    Xte = _np.random.randn(3, num_features).astype(_np.float32)
    yte_true = _np.random.randn(3).astype(_np.float32)  # For BarDistribution fitting

    # If BarDistribution is enabled, fit it first
    if w.use_bar_distribution:
        print("[SimplePFNSklearn] Fitting BarDistribution...")
        # Create multiple synthetic datasets for fitting
        train_datasets = [_np.random.randn(10, num_features).astype(_np.float32) for _ in range(5)]
        train_targets = [_np.random.randn(10).astype(_np.float32) for _ in range(5)]
        test_datasets = [_np.random.randn(5, num_features).astype(_np.float32) for _ in range(5)]
        test_targets = [_np.random.randn(5).astype(_np.float32) for _ in range(5)]
        
        w.fit_bar_distribution(train_datasets, train_targets, test_datasets, test_targets, max_batches=5)
        
        # Test different prediction types
        print("\n[SimplePFNSklearn] Testing different prediction types...")
        
        mode_preds = w.predict(Xtr, ytr, Xte, prediction_type="mode")
        print(f"Mode predictions shape: {mode_preds.shape}, values: {mode_preds}")
        
        mean_preds = w.predict(Xtr, ytr, Xte, prediction_type="mean") 
        print(f"Mean predictions shape: {mean_preds.shape}, values: {mean_preds}")
        
        samples = w.predict(Xtr, ytr, Xte, prediction_type="sample", num_samples=10)
        print(f"Sample predictions shape: {samples.shape}")
        print(f"First test point samples: {samples[:, 0]}")  # Show samples for first test point
        
    else:
        # Standard MSE prediction
        preds = w.predict(Xtr, ytr, Xte, prediction_type="point")
        print("[SimplePFNSklearn] Point prediction shape:", preds.shape)
        print("[SimplePFNSklearn] Point predictions:", preds)

    print("[SimplePFNSklearn] Test completed successfully!")
