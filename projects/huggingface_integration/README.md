# Huggingface Integration

Utilities for packaging Luxical embedders as Huggingface Transformers models and validating parity with the native Luxical pipeline. While the examples below use the Luxical One checkpoint, the wrapper is generic for any Luxical `Embedder` saved with format version 1.

## Layout

- `luxical_hf_wrapper.py` — Generic Huggingface `PreTrainedModel` wrapper around the Luxical `Embedder` plus an all-in-one CLI for export and verification.

## Prerequisites

1. Fetch a reference checkpoint. You may download this from our HuggingFace repo [here](https://huggingface.co/DatologyAI/luxical-one/resolve/main/luxical_one_rc4.npz).
2. Install runtime dependencies:
   - `luxical`
   - `transformers[torch]` (provides Transformers and PyTorch)
   - `safetensors` (installed transitively by Transformers, required for export)

## Exporting the Model

Produce a Huggingface directory (defaults shown). The CLI accepts any Luxical embedder `.npz`:

``` shell
uv run python luxical_hf_wrapper.py export \
  --checkpoint /tmp/my_saved_luxical_model.npz \
  --output-dir /tmp/my_saved_luxical_model_hf
```

The folder contains:
- `config.json` with `auto_map` so `AutoModel` can load the code
- `model.safetensors` containing all embedder weights and metadata
- `luxical_hf_wrapper.py` (the runtime model code for Hub/local use)
No separate `.npz` is used; exports are self-contained and safetensors-only.

## Verifying Parity

Run the verification command to confirm `AutoModel.from_pretrained(..., trust_remote_code=True)` matches `Embedder` outputs. Any compatible `.npz` may be used:

``` shell
uv run python luxical_hf_wrapper.py verify \
  --checkpoint /tmp/my_saved_luxical_model.npz \
  --export-dir /tmp/my_saved_luxical_model_hf
```

To consume the export directly:

``` python
from transformers import AutoModel
from pathlib import Path

# Load from the Huggingface Hub
hub_model = AutoModel.from_pretrained("datology/luxical-one", trust_remote_code=True)

# Load from a local export directory
local_dir = Path("/tmp/my_saved_luxical_model_hf").expanduser()
local_model = AutoModel.from_pretrained(local_dir, trust_remote_code=True)

emb = local_model(["Luxical integrates with Huggingface."]).embeddings
```
