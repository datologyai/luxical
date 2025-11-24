"""
Code for embedding data using the Arctic2M model as a teacher.

Uses MRL and 8-bit uniform scalar quantization.

NOTE: This code has been slightly optimized for throughput. Though the Python
multithreading is straightforward than a pure for-loop, it allows overlapping
tokenization work with GPU work and increases GPU utilization and overall throughput.

NOTE: You must `uv pip install xformers` to use the fast version of Arctic Embed 2M.
"""

import logging
from multiprocessing.pool import ThreadPool

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from luxical.teacher_embedder import EmbedderArctic2M

logger = logging.getLogger(__name__)

SOURCE_DATASET_URL = "s3://<your-bucket>/<your-prefix>/training_docs_fineweb/"
OUTPUT_DATASET_URL = "s3://<your-bucket>/<your-prefix>/training_docs_fineweb_arctic2m_mrl_8bit_quantized_scale_0dot3/"
MAX_SEQ_LEN = 512
EMBED_BATCH_SIZE = 256  # Number of texts per GPU batch.
QUANTIZATION_SCALE = 0.3
TABLE_CHUNK_SIZE = 4 * 1024  # Number of rows per arrow record batch.
OUT_SCHEMA = pa.schema(
    [
        ("document_id", pa.string()),
        ("embedding", pa.list_(pa.uint8(), EmbedderArctic2M.MRL_EMBEDDING_DIM)),
    ]
)


def read_row_count(path: str) -> int:
    with pq.ParquetFile(path, filesystem=s3) as pqf:
        return pqf.metadata.num_rows


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    s3 = fsspec.filesystem("s3", default_block_size=16 * 2**20)
    in_paths = [f"s3://{p}" for p in sorted(s3.ls(SOURCE_DATASET_URL))]

    # Load the tokenizer and model.
    embedder = EmbedderArctic2M(max_seq_len=MAX_SEQ_LEN)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    logger.info(f"Using device {device} and dtype {dtype}")
    embedder.to(device, dtype)

    # Count the number of rows in the dataset.
    with ThreadPool(processes=5) as pool:
        row_counts = pool.imap(read_row_count, in_paths)
        row_counts = list(
            tqdm(row_counts, total=len(in_paths), desc="Counting rows in input files")
        )
    total_row_count = sum(row_counts)
    logger.info(f"Total row count: {total_row_count:,d}")

    with (
        tqdm(
            total=total_row_count,
            desc="Embedding",
            unit="text",
            unit_scale=True,
        ) as pbar,
        logging_redirect_tqdm(),
    ):
        for in_path in in_paths:
            out_path = in_path.replace(SOURCE_DATASET_URL, OUTPUT_DATASET_URL)
            if s3.exists(out_path):
                logger.info(f"Skipping {in_path} because {out_path} already exists")
                with (
                    pq.ParquetFile(in_path, filesystem=s3) as pq_in,
                    pq.ParquetFile(out_path, filesystem=s3) as pq_out,
                ):
                    num_rows_in = pq_in.metadata.num_rows
                    num_rows_out = pq_out.metadata.num_rows
                if num_rows_in != num_rows_out:
                    raise ValueError(
                        f"Found output file {out_path} but number of rows "
                        f"({num_rows_out:,d}) does not match number of rows in "
                        f"input {in_path} ({num_rows_in:,d})"
                    )
                logger.info(f"Skipping {in_path} because {out_path} already exists")
                pbar.update(num_rows_in)
                continue
            logger.info(f"Embedding texts from {in_path} to {out_path}")
            with (
                pq.ParquetFile(in_path, filesystem=s3) as reader,
                pq.ParquetWriter(out_path, schema=OUT_SCHEMA, filesystem=s3) as writer,
            ):
                for chunk in reader.iter_batches(batch_size=TABLE_CHUNK_SIZE):
                    id_array = chunk["id"]
                    text_array = chunk["text"]
                    emb_ndarray = embedder.embed_texts(
                        text_array.to_pylist(),
                        is_query=False,
                        batch_size=EMBED_BATCH_SIZE,
                        mrl=True,
                        scalar_quantize_with_limit=QUANTIZATION_SCALE,
                        progress_bar=False,
                    )
                    fixed_size_list_array = pa.FixedSizeListArray.from_arrays(
                        pa.array(emb_ndarray.ravel()), emb_ndarray.shape[1]
                    )
                    rb = pa.RecordBatch.from_arrays(
                        [id_array, fixed_size_list_array],
                        schema=OUT_SCHEMA,
                    )
                    writer.write_batch(rb)
                    pbar.update(len(chunk))
            logger.info(f"Done embedding texts from {in_path} to {out_path}")
    logger.info("Done embedding.")
