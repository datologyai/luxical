import logging
import queue
import threading
from multiprocessing.pool import ThreadPool
from typing import Iterable, Iterator, Literal, Sequence

import numpy as np
import pyarrow as pa
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from luxical.dataset_abstractions import ManyParquetFileDataset
from luxical.misc_utils import (
    dequantize_8bit_uniform_scalar_quantized,
    pyarrow_fixed_size_list_array_to_numpy_ndarray,
)

logger = logging.getLogger(__name__)


def batch_index_generator(
    n: int, num_steps: int, batch_size: int, shuffle: bool = True, seed: int = 0
) -> Iterable[NDArray[np.int64]]:
    """Shuffles `n` indices and samples `batch_size` chunks (dropping last if incomplete)
    until `num_steps` batches have been emitted.
    """
    rng = np.random.default_rng(seed)
    if shuffle:
        idx = rng.permutation(n)
    else:
        idx = np.arange(n)
    i = 0
    for _ in range(num_steps):
        # Are we starting a new epoch?
        if i + batch_size > len(idx):
            if shuffle:
                rng.shuffle(idx)
            i = 0

        # Yield the next batch.
        batch_idx = idx[i : i + batch_size]
        yield batch_idx
        i += batch_size


def wsd_lr_schedule(
    step: int, total_steps: int, num_warmup: int, num_decay: int
) -> float:
    if step < num_warmup:
        return step / num_warmup
    if step > total_steps - num_decay:
        return (total_steps - step) / num_decay
    return 1.0


def equal_beta_adamw(
    params: Iterable[torch.nn.Parameter],
    lr: float,
    beta: float = 0.9,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> torch.optim.AdamW:
    """Creates an AdamW optimizer with a single beta value for all parameters.

    Uses different default arguments than the standard AdamW optimizer better suited
    to our training usecase.
    """
    optimizer = torch.optim.AdamW(
        params,
        lr=lr,
        betas=(beta, beta),
        eps=eps,
        weight_decay=weight_decay,
    )
    return optimizer


def remove_diagonal(x: torch.Tensor):
    """Removes the diagonal from a b x b shaped Gram matrix, leaving a b x (b - 1)
    shaped matrix with the trivial "always 1.0" self-similarity relationship removed.
    """
    assert x.ndim == 2 and x.size(0) == x.size(1), "x must be a square 2D tensor"
    mask = ~torch.eye(x.size(0), dtype=torch.bool, device=x.device)
    return x[mask].view(x.size(0), -1)


def contrastive_distillation_loss(
    student_embedding_batch: torch.Tensor,
    teacher_embedding_batch: torch.Tensor,
    loss_temperature: float,
) -> torch.Tensor:
    # Compute the gram matrices.
    gram_teacher = teacher_embedding_batch @ teacher_embedding_batch.T
    gram_student = student_embedding_batch @ student_embedding_batch.T

    # Remove the self-similarity diagonal from the gram matrices.
    gram_teacher = remove_diagonal(gram_teacher)
    gram_student = remove_diagonal(gram_student)

    # Loss is KL divergence when using scaled gram matrices as logits.
    loss = loss_temperature**2 * F.kl_div(
        input=F.log_softmax(gram_student / loss_temperature, dim=1),
        target=F.log_softmax(gram_teacher / loss_temperature, dim=1),
        log_target=True,
        reduction="batchmean",
    )
    return loss


#
# Dataloading from parquet files.
#


def dataloader(
    text_dataset: ManyParquetFileDataset,
    teacher_emb_dataset: ManyParquetFileDataset,
    teacher_emb_quantization_limit: float,
    batch_size: int,
    num_batches: int,
    num_examples_max_read_ahead: int = 16 * 1024,
    streaming_shuffle_buffer_size: int | Literal["auto"] = "auto",
) -> Iterator[tuple[list[str], torch.Tensor]]:
    """A dataloader that streams from two `ManyParquetFileDataset` objects."""
    if text_dataset.row_counts != teacher_emb_dataset.row_counts:
        raise ValueError(
            "Mismatch in texts and teacher embedding datasets."
            f"\n{text_dataset.row_counts=}\n{teacher_emb_dataset.row_counts=}"
        )
    assert len(text_dataset) == len(teacher_emb_dataset)

    # Announce the number of examples.
    num_examples_in_dataset = len(text_dataset)
    num_examples_per_epoch = num_examples_in_dataset - (
        num_examples_in_dataset % batch_size
    )
    logger.info(
        f"Dataloader will stream {num_batches:,d} batches of {batch_size:,d} examples, "
        f"totalling {num_batches * batch_size:,d} examples. "
        f"There will be {num_examples_per_epoch // batch_size} batches per epoch for a "
        f"total of {num_examples_per_epoch:,d} examples per epoch for "
        f"{num_batches / (num_examples_per_epoch // batch_size):.2f} epochs."
    )

    # Create a queue to hold the batches.
    read_ahead_num_batches = num_examples_max_read_ahead // batch_size
    q_maxsize = read_ahead_num_batches + 1
    q: queue.Queue[tuple[list[str], torch.Tensor] | None] = queue.Queue(
        maxsize=q_maxsize
    )
    logger.info(
        f"Dataloader queue size {q_maxsize}, for {read_ahead_num_batches} read-ahead batches."
    )

    def worker() -> None:
        try:
            yielded_count = 0
            repeat_count = 0
            while yielded_count < num_batches:
                logger.info(f"Starting dataloader repeat {repeat_count + 1}")
                text_batch_iter = text_dataset.stream_record_batches(
                    max_batch_size=batch_size, shuffle_files_with_seed=repeat_count
                )
                emb_batch_iter = teacher_emb_dataset.stream_record_batches(
                    max_batch_size=batch_size, shuffle_files_with_seed=repeat_count
                )
                shuffled_iter = arrow_streaming_shuffle(
                    record_batch_iterables=[text_batch_iter, emb_batch_iter],
                    output_batch_size=batch_size,
                    seed=repeat_count,
                    buffer_size=streaming_shuffle_buffer_size,
                )
                for [text_batch, emb_batch] in shuffled_iter:
                    assert len(text_batch) == len(emb_batch) == batch_size

                    # Pull out and match the ids.
                    assert text_batch["id"] == emb_batch["document_id"]

                    # Pull out and dequantize the embeddings.
                    quantized_teacher_emb_array = _ensure_not_chunked(
                        emb_batch["embedding"]
                    )
                    quantized_teacher_emb_ndarray = (
                        pyarrow_fixed_size_list_array_to_numpy_ndarray(
                            quantized_teacher_emb_array
                        )
                    )
                    teacher_emb_ndarray = dequantize_8bit_uniform_scalar_quantized(
                        quantized_teacher_emb_ndarray,
                        limit=teacher_emb_quantization_limit,
                    )
                    teacher_emb_tensor = torch.from_numpy(teacher_emb_ndarray)

                    # Pull out the texts.
                    texts = text_batch["text"].to_pylist()

                    # Put the result in the queue.
                    logger.debug(
                        f"Worker queueing batch {yielded_count + 1}/{num_batches}. "
                        f"Queue fullness before addition: {q.qsize()}/{q_maxsize}."
                    )
                    if q.full():
                        logger.debug("Dataloader queue full, worker is waiting.")
                    q.put((texts, teacher_emb_tensor))

                    yielded_count += 1
                    if yielded_count >= num_batches:
                        break
                repeat_count += 1
        except Exception:
            logger.exception("Terminating dataloading on dataloader worker error.")
            raise
        finally:
            # Sentinel to indicate that we're done.
            logger.debug("Queueing sentinel to indicate dataloader exit.")
            q.put(None)

    # Start the worker thread.
    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    # Yield items from the queue.
    while True:
        if q.empty():
            logger.debug("Dataloader queue empty, consumer is waiting.")
        item = q.get()
        if item is None:
            logger.debug("Read sentinel signal of last batch, dataloading is complete.")
            break
        texts, teacher_emb_tensor = item
        yield texts, teacher_emb_tensor


def arrow_streaming_shuffle(
    record_batch_iterables: Sequence[Iterable[pa.RecordBatch]],
    output_batch_size: int,
    buffer_size: int | Literal["auto"] = "auto",
    seed: int = 0,
) -> Iterable[Sequence[pa.RecordBatch]]:
    """Performs a streaming shuffle over one or more iterables of Arrow record batches.

    Parameters of note:
    - `output_batch_size`: The size of the output batches to take from the iterables.
    - `buffer_size`: The size of the buffer to use for the shuffle. Larger buffer sizes
        provide a more thorough shuffle, but require more memory and take longer to
        prefill at the start of streaming shuffling.
    """
    if buffer_size == "auto":
        buffer_size = 5 * output_batch_size
        logger.info(f"Using automatic buffer size {buffer_size:,d} (5x batch size)")
    rng = np.random.default_rng(seed)
    logger.info(
        f"Starting streaming shuffle of {len(record_batch_iterables)} iterables with "
        f"seed {seed}."
    )

    # Start iterating.
    iterators = [iter(rb_iterable) for rb_iterable in record_batch_iterables]

    # Fill the buffers initially.
    logger.info(f"Filling initial shuffle buffer up to {buffer_size} records.")
    with ThreadPool(processes=len(record_batch_iterables)) as pool:
        ran_out, buffer_tables = _take_together(iterators, buffer_size, pool)
        if buffer_tables is None:
            raise ValueError("No data was available from the record batch iterables.")
        logger.debug(
            f"Initial buffer filled with {_buffer_size(buffer_tables)} records. "
            f"Ran out of initial data: {ran_out}"
        )

        # Run the iteration over data.
        while True:
            # Seek to replenish.
            refill_target = buffer_size - _buffer_size(buffer_tables)
            if refill_target > 0 and not ran_out:
                logger.debug(
                    f"Refilling buffer by {refill_target:,d} records to achieve buffer "
                    f"size {buffer_size:,d}. Current buffer size is {_buffer_size(buffer_tables):,d}."
                )
                ran_out, addon_buffers = _take_together(iterators, refill_target, pool)
                if addon_buffers is not None:
                    buffer_tables = [
                        pa.concat_tables([b, b_new])
                        for b, b_new in zip(buffer_tables, addon_buffers)
                    ]
                logger.debug(
                    f"After refilling buffers, the new size is {_buffer_size(buffer_tables)}. Ran out of data: {ran_out}"
                )

            # Quit if we're out of data.
            current_buffer_size = _buffer_size(buffer_tables)
            if current_buffer_size < output_batch_size:
                assert ran_out
                logger.debug(
                    f"Buffer size ({current_buffer_size}) is smaller than batch size ({output_batch_size}) "
                    "and out of data. Streaming shuffle is complete."
                )
                break

            # Otherwise, take a random batch out of the buffers.
            idx = rng.choice(
                a=current_buffer_size, size=output_batch_size, replace=False
            )
            idx_kept = np.setdiff1d(np.arange(current_buffer_size), idx)
            out_tables = [buffer.take(idx) for buffer in buffer_tables]
            buffer_tables = [buffer.take(idx_kept) for buffer in buffer_tables]

            # The `out_tables` are tables with `output_batch_size` rows. We can
            # convert them to single record batches.
            yield [t.combine_chunks().to_batches()[0] for t in out_tables]


def _take_together(
    rb_iters: Sequence[Iterator[pa.RecordBatch]],
    n_or_more: int,
    pool: ThreadPool | None = None,
) -> tuple[bool, list[pa.Table] | None]:
    """Take at least n_or_more records from a sequence of record batch iterators."""
    assert n_or_more > 0, "n_or_more must be positive"
    logger.debug(
        f"Taking at least {n_or_more} records together from {len(rb_iters)} iterators."
    )
    current = 0
    chunk_lists = [[] for _ in range(len(rb_iters))]
    ran_out = False
    while current < n_or_more:
        if pool is None:
            chunks = [next(it, None) for it in rb_iters]
        else:
            chunks = pool.map(lambda it: next(it, None), rb_iters)
        if any(c is None for c in chunks):
            assert all(c is None for c in chunks), "All iters should run out together"
            logger.debug("Ran out of data in _take_together.")
            ran_out = True
            break
        assert chunks[0] is not None
        chunk_size = len(chunks[0])
        assert all(c is not None and len(c) == chunk_size for c in chunks)
        for chunk, chunk_list in zip(chunks, chunk_lists):
            chunk_list.append(chunk)
        current += chunk_size
    if len(chunk_lists[0]) == 0:
        tables = None
        assert ran_out
        logger.debug(f"No records were taken in _take_together. Ran out: {ran_out}")
    else:
        tables = [pa.Table.from_batches(chunk_list) for chunk_list in chunk_lists]
        num_records = len(tables[0])
        logger.debug(
            f"Finished taking records. Got {num_records} records. Ran out: {ran_out}"
        )
    return ran_out, tables


def _buffer_size(buffer_tables: list[pa.Table]) -> int:
    assert all(len(buffer) == len(buffer_tables[0]) for buffer in buffer_tables)
    return len(buffer_tables[0])


def _ensure_not_chunked(a: pa.Array | pa.ChunkedArray) -> pa.Array:
    if hasattr(a, "combine_chunks"):
        a = a.combine_chunks()
    return a
