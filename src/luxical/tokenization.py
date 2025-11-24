import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
from tokenizers import Tokenizer
from tqdm.auto import tqdm

from arrow_tokenize import ArrowTokenizer

logger = logging.getLogger(__name__)


def arrow_tokenizer_from_tokenizer(tokenizer: Tokenizer) -> ArrowTokenizer:
    return ArrowTokenizer(tokenizer.to_str())


def load_arrow_tokenizer_from_pretrained(tokenizer_id: str) -> ArrowTokenizer:
    return arrow_tokenizer_from_tokenizer(Tokenizer.from_pretrained(tokenizer_id))


def load_arrow_tokenizer_from_file(tokenizer_file: Path | str) -> ArrowTokenizer:
    tokenizer_state = Path(tokenizer_file).read_text()
    return ArrowTokenizer(tokenizer_state)


def arrow_tokenize_texts(
    texts: list[str],
    arrow_tokenizer: ArrowTokenizer,
    batch_size: int = 4096,
    add_special_tokens: bool = False,
    progress_bar: bool = True,
) -> pa.ChunkedArray:
    text_train_pa = pa.array(texts)
    token_chunks = []
    with (
        tqdm(
            total=len(text_train_pa),
            desc="Tokenizing",
            unit="seq",
            unit_scale=True,
            disable=not progress_bar,
        ) as pbar,
        tqdm(
            unit="token", unit_scale=True, position=1, disable=not progress_bar
        ) as pbar_tokens,
    ):
        for start in range(0, len(text_train_pa), batch_size):
            end = min(start + batch_size, len(text_train_pa))
            batch_text = text_train_pa[start:end]
            token_pa = arrow_tokenizer.tokenize(
                pa.array(batch_text), add_special_tokens=add_special_tokens
            )
            token_chunks.append(token_pa)
            total_tokens = pc.list_value_length(token_pa).sum().as_py()  # type: ignore[attr-defined]
            pbar.update(len(token_pa))
            pbar_tokens.update(total_tokens)
    token_array = pa.chunked_array(token_chunks)
    return token_array
