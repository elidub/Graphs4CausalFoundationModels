from __future__ import annotations
from typing import Optional, Dict, Any

import yaml
import torch
import numpy as np
import re
from pathlib import Path

# Robust import: try package-style, then relative, then add repo root to sys.path
try:
    from src.models.SimplePFN import SimplePFNRegressor
except Exception:
    try:
        from models.SimplePFN import SimplePFNRegressor
    except Exception:
        import sys
        repo_root = Path(__file__).resolve().parents[2]
        sys.path.append(str(repo_root))
        from src.models.SimplePFN import SimplePFNRegressor


class SimplePFNSklearn:
    """
    A small scikit-learn-like wrapper around the SimplePFN PyTorch model.

    Usage:
      wrapper = SimplePFNSklearn(config_path="path/to/config.yaml", checkpoint_path="model.pt")
      wrapper.load()                  # builds model and loads weights
      preds = wrapper.predict(X_train, y_train, X_test)

    Notes:
    - `fit` is a no-op placeholder (training is handled elsewhere); it returns self to
      satisfy the sklearn convention.
    - `predict` accepts numpy arrays or torch tensors. It accepts both batched inputs
      (B, N, F) and single-dataset inputs (N, F) which are treated as batch size 1.
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

    def load(self) -> "SimplePFNSklearn":
        """Load config (if provided), build the model and load checkpoint (if provided)."""
        if self.config_path:
            with open(self.config_path, "r") as f:
                cfg = yaml.safe_load(f)
            # model config lives under top-level 'model_config'
            mcfg = cfg.get("model_config", {}) if isinstance(cfg, dict) else {}
            # map expected keys
            self.model_kwargs = {
                "num_features": int(mcfg.get("num_features", 0)),
                "d_model": int(mcfg.get("d_model", 256)),
                "depth": int(mcfg.get("depth", 8)),
                "heads_feat": int(mcfg.get("heads_feat", 8)),
                "heads_samp": int(mcfg.get("heads_samp", 8)),
                "dropout": float(mcfg.get("dropout", 0.0)),
            }
        else:
            if self.verbose:
                print("[SimplePFNSklearn] No config_path provided; you must supply model kwargs manually before load().")

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

    def fit(self, X: Any = None, y: Any = None, **kwargs) -> "SimplePFNSklearn":
        """Placeholder fit to satisfy sklearn interface. Training should be done with the project's trainer.

        If checkpoint was provided `load()` should have been called before `fit`.
        """
        # no-op: we don't implement training here
        if self.verbose:
            print("[SimplePFNSklearn] fit() is a no-op placeholder. Use the training scripts to train the model.")
        return self

    def predict(self, X_train: Any, y_train: Any, X_test: Any) -> np.ndarray:
        """
        Run the PFN forward pass and return predictions as a numpy array.

        Inputs may be numpy arrays or torch tensors. Shapes accepted:
         - X_train: (N, F) or (B, N, F)
         - y_train: (N,) or (N,1) or (B, N) or (B,N,1)
         - X_test: (M, F) or (B, M, F)

        Output: numpy array of shape (B, M) or (M,) for batch size 1.
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call load() before predict().")

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
            preds = out["predictions"].cpu().numpy()

        # if batch size 1, return (M,) for convenience
        if preds.shape[0] == 1:
            return preds[0]
        return preds


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

    # prepare a tiny synthetic batch: 10 train samples, 3 test samples
    num_features = int(w.model.num_features)
    print(f"[SimplePFNSklearn] num_features = {num_features}")

    import numpy as _np
    Xtr = _np.random.randn(10, num_features).astype(_np.float32)
    ytr = _np.random.randn(10).astype(_np.float32)
    Xte = _np.random.randn(3, num_features).astype(_np.float32)

    preds = w.predict(Xtr, ytr, Xte)
    print("[SimplePFNSklearn] Prediction shape:", preds.shape)
    print("[SimplePFNSklearn] Predictions:", preds)
