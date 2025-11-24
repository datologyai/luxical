"""Huggingface integration for the Luxical embedder.

Export CLI usage:

```shell
# Export to Huggingface format.
uv run luxical_hf_wrapper.py export --checkpoint ~/Downloads/luxical_one_rc4.npz --output-dir /tmp/luxical-one

# Verify the export runs just like the native Luxical codepath.
uv run luxical_hf_wrapper.py verify --checkpoint ~/Downloads/luxical_one_rc4.npz --export-dir /tmp/luxical-one
```
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow as pa
import torch
from torch import Tensor
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import ModelOutput

from luxical.embedder import Embedder, _pack_int_dict, _unpack_int_dict
from luxical.sparse_to_dense_neural_nets import SparseToDenseEmbedder
from luxical.tokenization import ArrowTokenizer

DEFAULT_EMBEDDER_FILENAME = "luxical_one_embedder.npz"  # deprecated; no longer used


class LuxicalOneConfig(PretrainedConfig):
    """Configuration for the Luxical Huggingface wrapper.

    Generic for any Luxical `Embedder` serialized in format version 1.
    """

    model_type = "luxical-one"

    def __init__(
        self,
        *,
        max_ngram_length: int | None = None,
        embedding_dim: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.max_ngram_length = max_ngram_length
        self.embedding_dim = embedding_dim


@dataclass
class LuxicalOneModelOutput(ModelOutput):
    embeddings: Tensor


class LuxicalOneModel(PreTrainedModel):
    """Huggingface `PreTrainedModel` wrapper around a Luxical `Embedder`.

    Not tied to a specific checkpoint; reconstructs the `Embedder` from
    serialized state stored in the weights. Safetensors-only export.
    """

    config_class = LuxicalOneConfig

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):  # type: ignore[override]
        """Load model and reconstruct the Luxical embedder from safetensors.

        Keeps logic minimal and safetensors-only to avoid legacy branches.
        """
        model = super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        try:
            from transformers.utils import SAFE_WEIGHTS_NAME, cached_file
            from safetensors.torch import load_file as load_safetensors  # type: ignore
        except Exception:
            return model

        revision = kwargs.get("revision")
        cache_dir = kwargs.get("cache_dir")
        force_download = kwargs.get("force_download", False)
        proxies = kwargs.get("proxies")
        token = kwargs.get("token")
        local_files_only = kwargs.get("local_files_only", False)

        weight_path = None
        try:
            weight_path = cached_file(
                pretrained_model_name_or_path,
                SAFE_WEIGHTS_NAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                token=token,
                local_files_only=local_files_only,
            )
        except Exception:
            pass
        if weight_path is None:
            cand = Path(pretrained_model_name_or_path) / "model.safetensors"
            if cand.exists():
                weight_path = str(cand)

        if weight_path is not None:
            try:
                sd = load_safetensors(weight_path)
                model._embedder = _embedder_from_state_dict(sd)
                model._embedder_path = None
            except Exception:
                pass
        return model

    def __init__(
        self,
        config: LuxicalOneConfig,
        *,
        embedder: Embedder | None = None,
        embedder_path: str | Path | None = None,
    ) -> None:
        self._embedder: Embedder | None = embedder
        self._embedder_path: Path | None = (
            Path(embedder_path).resolve() if embedder_path is not None else None
        )
        super().__init__(config)

    def post_init(self) -> None:
        super().post_init()
        if self._embedder is not None:
            self.config.embedding_dim = self._embedder.embedding_dim
            self.config.max_ngram_length = self._embedder.max_ngram_length

    def forward(
        self,
        input_texts: Sequence[str] | pa.StringArray | None = None,
        *,
        batch_size: int = 4096,
        progress_bars: bool = False,
    ) -> LuxicalOneModelOutput:
        if input_texts is None:
            msg = "input_texts must be provided"
            raise ValueError(msg)
        embedder = self._ensure_embedder_loaded()
        embeddings_np = embedder(
            texts=input_texts,
            batch_size=batch_size,
            progress_bars=progress_bars,
        )
        embeddings = torch.from_numpy(embeddings_np)
        return LuxicalOneModelOutput(embeddings=embeddings)

    def save_pretrained(
        self,
        save_directory: str | Path,
        *args,
        **kwargs,
    ) -> tuple[OrderedDict[str, Tensor], LuxicalOneConfig]:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        # Prepare config with auto_map so AutoModel can import this module when
        # loading from a Hub/local repo with trust_remote_code=True.
        self.config.auto_map = {
            "AutoConfig": "luxical_hf_wrapper.LuxicalOneConfig",
            "AutoModel": "luxical_hf_wrapper.LuxicalOneModel",
        }
        # Persist the embedder inside a single Safetensors file.
        embedder = self._ensure_embedder_loaded()
        state_dict = _embedder_to_state_dict(embedder)
        from safetensors.torch import save_file as save_safetensors  # type: ignore

        save_safetensors(state_dict, str(save_path / "model.safetensors"))
        # Copy this module alongside to support remote code loading.
        import inspect
        import shutil

        module_src = Path(inspect.getsourcefile(LuxicalOneModel) or __file__).resolve()
        shutil.copyfile(module_src, save_path / "luxical_hf_wrapper.py")
        # Save config.json last.
        self.config.save_pretrained(save_path)
        return state_dict, self.config

    def load_state_dict(
        self, state_dict: OrderedDict[str, Tensor], strict: bool = True
    ):  # type: ignore[override]
        # Interpret the state dict as a serialized Luxical Embedder and rebuild it.
        try:
            self._embedder = _embedder_from_state_dict(state_dict)
            self._embedder_path = None
            # Update config fields if available
            self.config.embedding_dim = self._embedder.embedding_dim
            self.config.max_ngram_length = self._embedder.max_ngram_length
            return torch.nn.modules.module._IncompatibleKeys([], [])
        except KeyError:
            if strict:
                missing = list(state_dict.keys())
                raise NotImplementedError(
                    "LuxicalOneModel expected serialized embedder tensors; "
                    f"unexpected keys: {missing}"
                )
            return torch.nn.modules.module._IncompatibleKeys(
                [], list(state_dict.keys())
            )

    def get_input_embeddings(self) -> torch.nn.Module:
        msg = "LuxicalOneModel does not expose token embeddings."
        raise NotImplementedError(msg)

    def set_input_embeddings(self, value: torch.nn.Module) -> None:
        msg = "LuxicalOneModel does not support replacing token embeddings."
        raise NotImplementedError(msg)

    def resize_token_embeddings(self, *args, **kwargs) -> None:
        msg = "LuxicalOneModel does not use token embeddings."
        raise NotImplementedError(msg)

    def _ensure_embedder_loaded(self) -> Embedder:
        if self._embedder is not None:
            return self._embedder
        raise RuntimeError(
            "Luxical embedder is not initialized. Load this model via "
            "AutoModel/LuxicalOneModel.from_pretrained so weights can be "
            "decoded into an Embedder."
        )

    # No legacy file-based loader; all state lives in model.safetensors.


def export_embedder_to_huggingface_directory(
    embedder: Embedder,
    save_directory: str | Path,
    *,
    config_overrides: dict[str, object] | None = None,
) -> Path:
    save_path = Path(save_directory)
    config = LuxicalOneConfig(
        max_ngram_length=embedder.max_ngram_length,
        embedding_dim=embedder.embedding_dim,
        **(config_overrides or {}),
    )
    config.name_or_path = str(save_path.resolve())
    model = LuxicalOneModel(config=config, embedder=embedder)
    model.save_pretrained(save_path)
    return save_path


# No global Auto* registration; exports include `auto_map` in config.json.


def _embedder_to_state_dict(embedder: Embedder) -> OrderedDict[str, Tensor]:
    sd: "OrderedDict[str, Tensor]" = OrderedDict()
    # Version
    sd["embedder.version"] = torch.tensor([1], dtype=torch.long)
    # Tokenizer json bytes
    tok_bytes = np.frombuffer(
        embedder.tokenizer.to_str().encode("utf-8"), dtype=np.uint8
    )
    sd["embedder.tokenizer"] = torch.from_numpy(tok_bytes.copy())
    # Recognized ngrams
    sd["embedder.recognized_ngrams"] = torch.from_numpy(
        embedder.recognized_ngrams.astype(np.int64, copy=False)
    )
    # Hash map keys/values
    keys, vals = _unpack_int_dict(embedder.ngram_hash_to_ngram_idx)
    sd["embedder.ngram_keys"] = torch.from_numpy(keys.astype(np.int64, copy=False))
    sd["embedder.ngram_vals"] = torch.from_numpy(vals.astype(np.int64, copy=False))
    # IDF
    sd["embedder.idf_values"] = torch.from_numpy(
        embedder.idf_values.astype(np.float32, copy=False)
    )
    # Layers
    layers = embedder.bow_to_dense_embedder.layers
    sd["embedder.num_layers"] = torch.tensor([len(layers)], dtype=torch.long)
    for i, layer in enumerate(layers):
        sd[f"embedder.nn_layer_{i}"] = torch.from_numpy(
            layer.astype(np.float32, copy=False)
        )
    return sd


def _embedder_from_state_dict(state_dict: OrderedDict[str, Tensor]) -> Embedder:
    version = int(state_dict["embedder.version"][0].item())
    if version != 1:
        raise NotImplementedError(f"Unsupported embedder version: {version}")
    tok_bytes = bytes(
        state_dict["embedder.tokenizer"].cpu().numpy().astype(np.uint8).tolist()
    )
    tokenizer = ArrowTokenizer(tok_bytes.decode("utf-8"))
    recognized_ngrams = (
        state_dict["embedder.recognized_ngrams"]
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )
    keys = state_dict["embedder.ngram_keys"].cpu().numpy().astype(np.int64, copy=False)
    vals = state_dict["embedder.ngram_vals"].cpu().numpy().astype(np.int64, copy=False)
    ngram_map = _pack_int_dict(keys, vals)
    idf_values = (
        state_dict["embedder.idf_values"].cpu().numpy().astype(np.float32, copy=False)
    )
    num_layers = int(state_dict["embedder.num_layers"][0].item())
    layers = [
        state_dict[f"embedder.nn_layer_{i}"]
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
        for i in range(num_layers)
    ]
    s2d = SparseToDenseEmbedder(layers=layers)
    return Embedder(
        tokenizer=tokenizer,
        recognized_ngrams=recognized_ngrams,
        ngram_hash_to_ngram_idx=ngram_map,
        idf_values=idf_values,
        bow_to_dense_embedder=s2d,
    )


def _parse_cli_args() -> tuple[str, dict[str, object]]:
    import argparse

    parser = argparse.ArgumentParser(
        description="Luxical One Huggingface wrapper: export and verify utilities.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_export = sub.add_parser(
        "export",
        help="Export a HF-formatted directory from a Luxical embedder .npz checkpoint",
    )
    p_export.add_argument(
        "--checkpoint",
        type=str,
        help="Path to Luxical embedder .npz checkpoint",
    )
    p_export.add_argument(
        "--output-dir",
        type=str,
        help="Directory to write the Huggingface-formatted model",
    )

    p_verify = sub.add_parser(
        "verify", help="Verify HF-loaded model matches native Embedder outputs"
    )
    p_verify.add_argument(
        "--checkpoint",
        type=str,
        help="Path to Luxical embedder .npz checkpoint",
    )
    p_verify.add_argument(
        "--export-dir",
        type=str,
        help="HF directory to create/use for verification",
    )

    args = parser.parse_args()
    return args.cmd, vars(args)


def _sample_texts() -> list[str]:
    return [
        "Luxical embeddings make tf-idf sparkle.",
        "This sentence tests the Huggingface wrapper path.",
        "Short.",
    ]


def _cmd_export(checkpoint: str, output_dir: str) -> None:
    ckpt_path = Path(checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}.")
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    embedder = Embedder.load(ckpt_path)
    export_embedder_to_huggingface_directory(embedder, out_dir)
    print(f"Huggingface directory written to {out_dir}")


def _cmd_verify(checkpoint: str, export_dir: str) -> None:
    ckpt_path = Path(checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}.")
    exp_dir = Path(export_dir).expanduser().resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)

    texts = _sample_texts()
    embedder = Embedder.load(ckpt_path)
    ref = embedder(texts)

    export_embedder_to_huggingface_directory(embedder, exp_dir)
    # Load using AutoModel so this mirrors user experience, with remote code.
    from transformers import AutoModel  # local import to keep top-level light

    model = AutoModel.from_pretrained(exp_dir, trust_remote_code=True)
    model.eval()
    with torch.inference_mode():
        out = model(texts).embeddings.cpu().numpy()

    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)
    print("Verification succeeded: Huggingface model matches embedder output.")


if __name__ == "__main__":
    cmd, kv = _parse_cli_args()
    if cmd == "export":
        _cmd_export(checkpoint=str(kv["checkpoint"]), output_dir=str(kv["output_dir"]))
    elif cmd == "verify":
        _cmd_verify(checkpoint=str(kv["checkpoint"]), export_dir=str(kv["export_dir"]))
    else:
        raise SystemExit(f"Unknown command: {cmd}")
