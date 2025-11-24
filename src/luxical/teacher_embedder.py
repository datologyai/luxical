"""
Code for embedding data using the Arctic2M model as a teacher.

Uses MRL and 8-bit uniform scalar quantization.

NOTE: This code has been slightly optimized for throughput at the expense of readability.
In particular, we use multithreading to overlap tokenization work with GPU work, which
increases GPU utilization and overall throughput.
"""

import logging
from multiprocessing.pool import ThreadPool
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from luxical.misc_utils import fast_8bit_uniform_scalar_quantize

NDArrayOfFloat = NDArray[np.floating[Any]]
NDArrayOfUint8 = NDArray[np.uint8]
logger = logging.getLogger(__name__)


BATCH_SIZE = 256
MAX_SEQ_LEN = 512
QUANTIZATION_SCALE = 0.3


def first_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pools the hidden states by selecting the first non-padding token representation
    for each sequence."""
    batch_size = last_hidden_states.shape[0]
    row = torch.arange(batch_size, device=last_hidden_states.device)
    col = attention_mask.argmax(dim=1)  # position of the first non-padding token
    return last_hidden_states[row, col]


class EmbedderArctic2M:
    HF_MODEL_ID: str = "Snowflake/snowflake-arctic-embed-m-v2.0"
    QUERY_PREFIX: str = "query: "
    DOC_PREFIX: str = ""
    EMBEDDING_DIM: int = 768
    MRL_EMBEDDING_DIM: int = 256

    def __init__(
        self, max_seq_len: int = 8192, use_memory_efficient_attention: bool = True
    ):
        logger.info(
            f"Loading tokenizer and model model from HuggingFace: `{self.HF_MODEL_ID}`"
        )
        self.model = AutoModel.from_pretrained(
            self.HF_MODEL_ID,
            trust_remote_code=True,
            unpad_inputs=use_memory_efficient_attention,
            use_memory_efficient_attention=use_memory_efficient_attention,
        )
        self.device: str | torch.device = "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.HF_MODEL_ID)
        assert isinstance(tokenizer, PreTrainedTokenizerFast)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def to(self, device: str | torch.device, dtype: torch.dtype | None = None) -> None:
        self.device = device
        self.model = self.model.to(device, dtype=dtype)

    def _tokenize(
        self, texts: Sequence[str], prefix: str, max_seq_len: int | None
    ) -> dict[str, Tensor]:
        prefixed_texts = [f"{prefix}{text}" for text in texts]
        inputs: dict[str, Tensor] = self.tokenizer(  # type: ignore
            prefixed_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
        return inputs

    @torch.inference_mode()
    def _embed_batch(
        self,
        inputs: dict[str, Tensor],
        mrl: bool = False,
        scalar_quantize_with_limit: float | None = None,
    ) -> NDArrayOfFloat | NDArrayOfUint8:
        outputs = self.model(**inputs)
        seq_of_vec = outputs.last_hidden_state
        vec = first_token_pool(seq_of_vec, inputs["attention_mask"])
        if mrl:
            vec = vec[:, : self.MRL_EMBEDDING_DIM]
        vec = F.normalize(vec, dim=-1)
        vec = vec.float().cpu().numpy()
        if scalar_quantize_with_limit is not None:
            vec = fast_8bit_uniform_scalar_quantize(vec, scalar_quantize_with_limit)
        return vec

    def embed_texts(
        self,
        texts: Sequence[str],
        is_query: bool,
        batch_size: int,
        mrl: bool = False,
        scalar_quantize_with_limit: float | None = None,
        progress_bar: bool = True,
    ) -> NDArrayOfFloat | NDArrayOfUint8:
        n = len(texts)
        out_dim = self.MRL_EMBEDDING_DIM if mrl else self.EMBEDDING_DIM
        if scalar_quantize_with_limit is not None:
            embeddings = np.zeros((n, out_dim), dtype=np.uint8)
        else:
            embeddings = np.zeros((n, out_dim), dtype=np.float32)
        text_batch_iter = (texts[i : i + batch_size] for i in range(0, n, batch_size))
        out_start = 0
        with (
            ThreadPool(1) as tokenizer_pool,
            tqdm(
                total=n, desc="Embedding", unit="text", disable=not progress_bar
            ) as pbar,
        ):
            # Tokenize concurrently so that we can do the next batch while the previous
            # batch is being embedded.
            input_iter = tokenizer_pool.imap(
                lambda batch: self._tokenize(
                    texts=batch,
                    prefix=self.QUERY_PREFIX if is_query else self.DOC_PREFIX,
                    max_seq_len=self.max_seq_len,
                ),
                text_batch_iter,
            )
            for inputs in input_iter:
                batch = self._embed_batch(
                    inputs,
                    mrl=mrl,
                    scalar_quantize_with_limit=scalar_quantize_with_limit,
                )
                embeddings[out_start : out_start + len(batch)] = batch
                out_start += len(batch)
                pbar.update(len(batch))
        return embeddings
