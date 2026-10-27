"""Optional local Torch Hub code paths; default model/weight behavior is unchanged."""
import os
from pathlib import Path
import torch


def load_hub_model(repo, model, *, env_var, **kwargs):
    local = os.environ.get(env_var)
    if local:
        directory = Path(local).expanduser().resolve()
        if not (directory / "hubconf.py").is_file():
            raise FileNotFoundError(f"{env_var} must contain hubconf.py: {directory}")
        repo = str(directory)
        kwargs["source"] = "local"
    return torch.hub.load(repo, model, **kwargs)
