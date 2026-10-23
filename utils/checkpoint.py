"""Load weights while discarding only the old THOP profiling counters."""
import torch


def load_weights(model, path, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=True)
    state = {k: v for k, v in state.items()
             if k.rsplit(".", 1)[-1] not in {"total_ops", "total_params"}}
    model.load_state_dict(state, strict=True)
