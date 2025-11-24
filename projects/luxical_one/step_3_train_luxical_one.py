import logging
from contextlib import contextmanager
from os.path import join
from time import perf_counter

import fsspec
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from uniplot import plot

from luxical.csr_matrix_utils import csr_matrix_to_torch
from luxical.embedder import initialize_embedder_from_ngram_summary
from luxical.misc_utils import find_project_root
from luxical.ngrams import SpaceSavingNgramSummary
from luxical.sparse_to_dense_neural_nets import SparseToDenseEmbedder
from luxical.tokenization import load_arrow_tokenizer_from_pretrained
from luxical.training import (
    ManyParquetFileDataset,
    contrastive_distillation_loss,
    dataloader,
    equal_beta_adamw,
    wsd_lr_schedule,
)

# Configure logging.
logger = logging.getLogger(__name__)

# Define data to use.
DATA_DIR = find_project_root() / "data"
TOKENIZER_ID = "google-bert/bert-base-uncased"
NGRAM_SUMMARY_FILEPATH = DATA_DIR / "ngram_summary_5mdoc.npz"
TEXT_DIR_PATH = str(DATA_DIR / "training_docs_fineweb")
TEACHER_EMBEDDING_DIR_PATH = str(
    DATA_DIR / "training_docs_fineweb_arctic2m_mrl_8bit_quantized_scale_0dot3"
)

# Pull out the quantization limit from the embedding file name.
TEACHER_EMB_QUANTIZATION_LIMIT = float(
    str(TEACHER_EMBEDDING_DIR_PATH).rsplit("_quantized_scale_")[-1].replace("dot", ".")
)


@contextmanager
def time_lines(timing_dict: dict[str, float], name: str):
    """Context manager for timing sections of code for debug logging of perf analysis."""
    start = perf_counter()
    try:
        yield None
    finally:
        end = perf_counter()
        duration = end - start
        timing_dict[name] = duration


def main(
    num_epochs: float = 3,
    batch_size: int = 12 * 1024,
    loss_temperature: float = 3.0,
    learning_rate: float = 1e-2,
    warmup_fraction: float = 0.05,
    decay_fraction: float = 0.1,
    sparse_to_dense_embedder_dims: tuple[int, ...] = (96, 3072, 3072, 192),
    min_ngram_count_multiple: float = 8.0,
    out_file_name: str = "luxical_one.npz",
):
    # Create record batch iterators from these parquet directories.
    logger.info(
        f"Reading text from {TEXT_DIR_PATH} and teacher embeddings from {TEACHER_EMBEDDING_DIR_PATH}"
    )
    protocol = "s3" if TEXT_DIR_PATH.startswith("s3://") else "file"
    protocol_prefix = "" if protocol == "file" else protocol + "://"
    fs = fsspec.filesystem(protocol)
    text_filepaths = [
        protocol_prefix + x for x in sorted(fs.glob(join(TEXT_DIR_PATH, "*.parquet")))
    ]
    emb_filepaths = [
        protocol_prefix + x
        for x in sorted(fs.glob(join(TEACHER_EMBEDDING_DIR_PATH, "*.parquet")))
    ]
    logger.info(
        f"Found {len(text_filepaths)} text files and {len(emb_filepaths)} embedding files"
    )
    text_dataset = ManyParquetFileDataset(text_filepaths)
    emb_dataset = ManyParquetFileDataset(emb_filepaths)
    if text_dataset.row_counts != emb_dataset.row_counts:
        raise ValueError("Mismatch in texts and embeddings")

    # Initialize the embedder.
    logger.info(f"Loading ngram summary from {NGRAM_SUMMARY_FILEPATH}")
    ngram_summary = SpaceSavingNgramSummary.load_npz(NGRAM_SUMMARY_FILEPATH)
    logger.info(f"Loading tokenizer {TOKENIZER_ID}")
    tokenizer = load_arrow_tokenizer_from_pretrained(TOKENIZER_ID)
    embedder = initialize_embedder_from_ngram_summary(
        ngram_summary=ngram_summary,
        tokenizer=tokenizer,
        sparse_to_dense_embedder_dims=sparse_to_dense_embedder_dims,
        min_ngram_count_multiple=min_ngram_count_multiple,
    )
    del ngram_summary  # Get back RAM by un-loading the ngram summary.

    # Create the trainable torch embedder.
    bow_to_dense_torch = embedder.bow_to_dense_embedder.to_torch()

    # Create a single-beta AdamW optimizer for the net.
    optimizer = equal_beta_adamw(
        params=bow_to_dense_torch.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )

    # Configure training.
    num_examples_per_epoch = batch_size * (len(text_dataset) // batch_size)
    num_examples = int(num_epochs * num_examples_per_epoch)
    num_steps: int = num_examples // batch_size
    num_warmup = int(num_steps * warmup_fraction)
    num_decay = int(num_steps * decay_fraction)

    dl = dataloader(
        text_dataset,
        emb_dataset,
        teacher_emb_quantization_limit=TEACHER_EMB_QUANTIZATION_LIMIT,
        batch_size=batch_size,
        num_batches=num_steps,
        streaming_shuffle_buffer_size=8 * batch_size,
        num_examples_max_read_ahead=32 * batch_size,
    )
    losses: list[float] = []
    lrs: list[float] = []
    try:
        with tqdm(total=num_examples, unit="example", unit_scale=True) as pbar:
            for step, (text_batch, teacher_emb_batch) in enumerate(dl):
                step += 1  # 1-index the step.
                if step == 1:
                    pbar.reset()  # Reset progress to account for dataload pre-fetch.

                # Set the learning rate.
                current_lr = learning_rate * wsd_lr_schedule(
                    step=step,
                    total_steps=num_steps,
                    num_warmup=num_warmup,
                    num_decay=num_decay,
                )
                lrs.append(current_lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr
                timing_dict = {}
                with time_lines(timing_dict, "text to tokens"):
                    tokens = embedder.tokenize(text_batch)
                with time_lines(timing_dict, "tokens to BoW"):
                    bow = embedder.bow_from_tokens(tokens)
                with time_lines(timing_dict, "BoW to TF-IDF"):
                    tfidf_torch = csr_matrix_to_torch(embedder.tfidf_from_bow(bow))
                with time_lines(timing_dict, "forward pass"):
                    student_emb_batch = bow_to_dense_torch(tfidf_torch)
                with time_lines(timing_dict, "contrastive loss"):
                    loss = contrastive_distillation_loss(
                        student_embedding_batch=student_emb_batch,
                        teacher_embedding_batch=teacher_emb_batch,
                        loss_temperature=loss_temperature,
                    )
                with time_lines(timing_dict, "backwards pass"):
                    loss.backward()
                with time_lines(timing_dict, "optimizer step"):
                    optimizer.step()
                    optimizer.zero_grad()
                loss_float = loss.item()
                losses.append(loss_float)
                logger.info(f"Loss: {loss_float:.6f}")
                total_time = sum(timing_dict.values())
                timing_msg = " | ".join(
                    f"{k}: {v:.2f}s" for k, v in timing_dict.items()
                )
                logger.info(f"Step {step:04d} took {total_time:.2f}s. {timing_msg}")
                pbar.update(batch_size)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user, finishing early.")

    # Edge case: No training.
    if len(losses) == 0:
        logger.warning("No losses recorded, so no recap or saving the embedder.")
        return

    # Plot the losses and learning rates.
    logger.info(f"Training complete. Final loss: {losses[-1]:.6f}")
    plot_width = 100
    plot_height = 30
    plot(
        ys=lrs,
        title="Learning Rate",
        lines=True,
        x_unit="steps",
        width=plot_width,
        height=plot_height,
    )
    print("\n\n")
    plot(
        ys=losses,
        title="Loss",
        lines=False,
        x_unit="steps",
        width=plot_width,
        height=plot_height,
    )

    # Save the results.
    new_embedder = SparseToDenseEmbedder.from_torch(bow_to_dense_torch)
    embedder = embedder.replace_sparse_to_dense_embedder(new_embedder)
    out_file_path = DATA_DIR / out_file_name
    logger.info(f"Saving trained embedder to {out_file_path}")
    embedder.save(path=out_file_path)


if __name__ == "__main__":
    # Configure logging.
    # Set basicConfig to INFO level for all other modules.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )
    # Set the logger for this file to DEBUG, so we see our debug messages.
    logging.getLogger(__name__).setLevel(logging.DEBUG)
    logging.getLogger("luxical.training").setLevel(logging.DEBUG)
    with logging_redirect_tqdm():
        main()
