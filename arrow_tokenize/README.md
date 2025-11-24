# Rust-based Python Tokenizer to Arrow

This project provides a tiny Python extension written in Rust for tokenizing text data directly into Arrow arrays. It uses the tokenizers library from Hugging Face and serializes the output directly into a pyarrow.LargeListArray without exposing any `Encoding` objects to the Python interpreter.

The primary benefit of this approach is memory efficiency. By avoiding the creation of intermediate Python lists of integers, it significantly reduces the overhead on the Python garbage collector, making it ideal for tokenizing very large datasets. This can provide a major speedup for large batch tokenization jobs, which otherwise can get bottlenecked on Python GC.

## Features

- Very simple API
- Provides a zero-copy data transfer of resulting Arrow arrays

## Setup and Installation

### Prerequisites

- Install Rust
- Install Python

### Installation Steps

`uv run maturin develop`

## Arrow-based Tokenization

Luxical bundles in the `arrow-tokenize` Rust extension package, which exposes `ArrowTokenizer`, a fast tokenizer abstraction that operates on Arrow arrays and returns Arrow arrays. This avoids Python‑level overhead during large batch tokenization. Compared to the Python API of the `tokenizers` library, `arrow-tokenize` dramatically reduces pressure on the Python garbage collector and can deliver substantial speedups in bulk tokenization.

Key Python API functions:
- `load_arrow_tokenizer_from_pretrained(tokenizer_id: str) -> ArrowTokenizer`
- `load_arrow_tokenizer_from_file(tokenizer_file: Path | str) -> ArrowTokenizer`
- `arrow_tokenize_texts(texts: list[str], arrow_tokenizer: ArrowTokenizer, *, batch_size=4096, add_special_tokens=False, progress_bar=True) -> pa.ChunkedArray`

Minimal example:

```python
from luxical.tokenization import (
    load_arrow_tokenizer_from_pretrained,
    arrow_tokenize_texts,
)

tok = load_arrow_tokenizer_from_pretrained("google-bert/bert-base-uncased")
chunks = arrow_tokenize_texts([
    "hello world",
    "lexical embeddings are fast",
], tok, batch_size=2, add_special_tokens=False, progress_bar=False)
```

## Release Notes

### v1.0.0 - 2025-09-22
- Initial release. Intended to be the only release until we need to bump dependency versions.
