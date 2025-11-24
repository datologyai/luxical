"""
Summarize the approximate counts of the most common 5-grams in the training documents.

TODO: Debug why there appears to be a memory-related crash for very large jobs. It
appears that this may be related to the S3-streaming tokenization approach, since memory
growth does not appear to be as large when using pre-tokenized documents from local
disk. For now we just will keep the limit at 5M docs and 200M ngrams.
"""

import logging
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import fsspec
import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import psutil
import os

from luxical.ngrams import space_saving_ngram_summary
from luxical.tokenization import ArrowTokenizer, load_arrow_tokenizer_from_pretrained

logger = logging.getLogger(__name__)

SOURCE_DATASET_URL = "s3://<your-bucket>/<your-prefix>/training_docs_fineweb/"
OUTPUT_DIR_URL = "s3://<your-bucket>/<your-prefix>/ngram_summaries/"
NGRAM_SIZE = 5
NUM_TOP_NGRAMS = 200_000_000
TOKENIZER_ID = "google-bert/bert-base-uncased"
READ_CHUNK_SIZE = 1024
LIMIT_DOCS = 5_000_000


def stream_tokenized_docs(
    arrow_tokenizer: ArrowTokenizer,
    in_paths: list[str],
) -> Iterable[NDArray[np.uint32]]:
    # Count the number of rows in the dataset.
    def read_row_count(path: str) -> int:
        with pq.ParquetFile(path) as pqf:
            return pqf.metadata.num_rows

    with ThreadPool(processes=10) as pool:
        row_counts = pool.imap(read_row_count, in_paths)
        row_counts = list(
            tqdm(row_counts, total=len(in_paths), desc="Counting rows in input files")
        )
    total_row_count = sum(row_counts)
    logger.info(f"Total row count: {total_row_count:,d}")
    total_row_count = min(total_row_count, LIMIT_DOCS)
    with tqdm(
        total=total_row_count, desc="Processing documents", unit="doc", unit_scale=True
    ) as pbar:
        for in_path in in_paths:
            table = pq.read_table(in_path)
            for chunk in table.to_batches():
                text_array = chunk["text"]
                token_array = arrow_tokenizer.tokenize(text_array)
                token_ndarrays = token_array.to_numpy(zero_copy_only=False)
                for doc_tokens in token_ndarrays:
                    yield doc_tokens
                    pbar.update()
                if pbar.n >= total_row_count:
                    return


def memory_monitoring_stream(
    stream: Iterable[NDArray[np.uint32]],
) -> Iterable[NDArray[np.uint32]]:
    process = psutil.Process(os.getpid())
    for i, item in enumerate(stream):
        yield item
        if i > 0 and i % 5_000 == 0:
            mem_info = process.memory_info()
            logger.info(
                f"Memory usage after {i:,} docs: {mem_info.rss / 1024**3:.2f} GiB"
            )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    s3 = fsspec.filesystem("s3")
    in_paths = [f"s3://{p}" for p in sorted(s3.ls(SOURCE_DATASET_URL))]

    # Load the tokenizer.
    arrow_tokenizer = load_arrow_tokenizer_from_pretrained(TOKENIZER_ID)

    # Stream the tokenized documents.
    tokenized_doc_stream = stream_tokenized_docs(arrow_tokenizer, in_paths)
    tokenized_doc_stream = memory_monitoring_stream(tokenized_doc_stream)

    # Compute the unigram bow matrix.
    with logging_redirect_tqdm():
        ss_summary = space_saving_ngram_summary(
            tokenized_doc_stream,
            max_ngram_length=NGRAM_SIZE,
            num_top_items=NUM_TOP_NGRAMS,
        )
    logger.info(
        f"Summarized {ss_summary.total_ngrams_seen:,} ngrams (n={NGRAM_SIZE}), skipping "
        f"{ss_summary.hash_collisions_skipped:,} due to hash collisions."
    )
    with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        temp_path = Path(temp_dir) / "ngram_summary_5mdoc.npz"
        logger.info(f"Saving the ngram summary to {temp_path}...")
        ss_summary.save_npz(temp_path)
        logger.info(f"Uploading the ngram summary to {OUTPUT_DIR_URL}...")
        s3.put(temp_path, OUTPUT_DIR_URL)
    logger.info(f"Successfully saved the ngram summary to {OUTPUT_DIR_URL}.")
