from __future__ import annotations

import hashlib
import logging
import time
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Iterator, Sequence

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ManyParquetFileDataset:
    """A dataset of many Parquet files that can be streamed."""

    def __init__(self, pq_paths: Sequence[str] | Sequence[Path]) -> None:
        if len(pq_paths) == 0:
            raise ValueError("No paths provided to ManyParquetFileDataset")

        # Convert to list of strings.
        self.pq_paths = [str(p) for p in pq_paths]
        del pq_paths

        logger.info(
            f"Initializing dataset {self.hashed_id} from {self.pq_paths[0]}, ..."
        )

        # Determine the filesystem of the parquet files.
        if self.pq_paths[0].startswith("s3://"):
            if not all(p.startswith("s3://") for p in self.pq_paths):
                raise NotImplementedError("Mixed S3 and non-S3 paths not supported")
            self.protocol = "s3"
        else:
            self.protocol = "file"
        filesystem = fsspec.filesystem(self.protocol)

        # Count the number of rows in the dataset.
        def read_row_count(path: str) -> tuple[str, int]:
            with pq.ParquetFile(path, filesystem=filesystem) as pqf:
                return path, pqf.metadata.num_rows

        with ThreadPool(processes=10) as pool:
            row_count_tuples = pool.imap_unordered(read_row_count, self.pq_paths)
            row_count_map = dict(
                tqdm(
                    row_count_tuples,
                    total=len(self.pq_paths),
                    desc="Counting rows in input files",
                )
            )
            self.row_counts = [row_count_map[path] for path in self.pq_paths]

        # Announce completion of initialization.
        logger.info(
            f"Dataset {self.hashed_id} initialized. Total row count: {len(self):,d}"
        )

    def __len__(self) -> int:
        return sum(self.row_counts)

    @property
    def hashed_id(self) -> str:
        return hashlib.sha256(str(self.pq_paths).encode()).hexdigest()[:4]

    def stream_record_batches(
        self, max_batch_size: int = 4096, shuffle_files_with_seed: int | None = None
    ) -> Iterator[pa.RecordBatch]:
        # Optionally shuffle the files.
        if shuffle_files_with_seed is None:
            idx = np.arange(len(self.pq_paths))
        else:
            logger.info(
                f"Dataset {self.hashed_id} shuffling {len(self.pq_paths)} files with seed {shuffle_files_with_seed}."
            )
            rng = np.random.default_rng(seed=shuffle_files_with_seed)
            idx = rng.permutation(len(self.pq_paths))
        pq_paths = [self.pq_paths[i] for i in idx]

        # Iterate through the files by row group.
        total_bytes_yielded = 0
        start_time = time.time()
        filesystem = fsspec.filesystem(self.protocol)
        for i, path in enumerate(pq_paths):
            logger.info(f"Streaming from file {i + 1}/{len(pq_paths)}: {path}")
            with pq.ParquetFile(path, filesystem=filesystem) as pqf:
                for batch in pqf.iter_batches(batch_size=max_batch_size):
                    # Track throughput statistics.
                    total_bytes_yielded += batch.nbytes
                    elapsed_time = time.time() - start_time
                    mib_yielded = total_bytes_yielded / (1024 * 1024)
                    throughput = mib_yielded / elapsed_time
                    logger.debug(
                        f"Dataset {self.hashed_id} yielding record batch. Average streaming throughput: {throughput:.2f} MiB/s"
                    )

                    # Yield the batch.
                    yield batch
