from typing import Iterator

import numpy as np
import pyarrow as pa
import torch

from luxical.training import (
    arrow_streaming_shuffle,
    batch_index_generator,
    contrastive_distillation_loss,
    remove_diagonal,
    wsd_lr_schedule,
)


def test_batch_index_generator():
    """Test batch index generation with shuffling and epochs."""
    n = 100
    batch_size = 10
    num_steps = 15  # 1.5 epochs

    batches = list(
        batch_index_generator(n, num_steps, batch_size, shuffle=False, seed=42)
    )

    assert len(batches) == num_steps

    # Check batch sizes
    for batch in batches:
        assert len(batch) == batch_size

    # Without shuffle, first batch should be [0, 1, ..., 9]
    np.testing.assert_array_equal(batches[0], np.arange(10))

    # Test with shuffling
    batches_shuffled = list(
        batch_index_generator(n, num_steps, batch_size, shuffle=True, seed=42)
    )
    assert len(batches_shuffled) == num_steps
    assert not np.array_equal(batches_shuffled[0], np.arange(10))

    # With same seed, should be deterministic
    batches_shuffled2 = list(
        batch_index_generator(n, num_steps, batch_size, shuffle=True, seed=42)
    )
    for b1, b2 in zip(batches_shuffled, batches_shuffled2):
        np.testing.assert_array_equal(b1, b2)


def test_wsd_lr_schedule():
    """Test warmup-stable-decay learning rate schedule."""
    total_steps = 1000
    num_warmup = 100
    num_decay = 200

    # Test warmup phase
    assert wsd_lr_schedule(0, total_steps, num_warmup, num_decay) == 0.0
    assert wsd_lr_schedule(50, total_steps, num_warmup, num_decay) == 0.5
    assert wsd_lr_schedule(100, total_steps, num_warmup, num_decay) == 1.0

    # Test stable phase
    assert wsd_lr_schedule(500, total_steps, num_warmup, num_decay) == 1.0
    assert wsd_lr_schedule(799, total_steps, num_warmup, num_decay) == 1.0

    # Test decay phase (starts at step 800 = 1000 - 200)
    assert wsd_lr_schedule(800, total_steps, num_warmup, num_decay) == 1.0
    assert wsd_lr_schedule(900, total_steps, num_warmup, num_decay) == 0.5
    assert wsd_lr_schedule(1000, total_steps, num_warmup, num_decay) == 0.0


def test_remove_diagonal():
    """Test diagonal removal from square matrices."""
    # 3x3 matrix
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)

    result = remove_diagonal(x)

    # Should be 3x2 matrix with diagonal removed
    expected = torch.tensor([[2, 3], [4, 6], [7, 8]], dtype=torch.float32)

    torch.testing.assert_close(result, expected)
    assert result.shape == (3, 2)


def test_contrastive_distillation_loss():
    """Test contrastive distillation loss computation."""
    batch_size = 4
    embedding_dim = 8

    # Create some normalized embeddings, which will make this numerically stable
    rng = torch.Generator()
    rng.manual_seed(0)
    student_emb = torch.randn(batch_size, embedding_dim, generator=rng)
    student_emb = torch.nn.functional.normalize(student_emb, dim=1)
    teacher_emb = torch.randn(batch_size, embedding_dim, generator=rng)
    teacher_emb = torch.nn.functional.normalize(teacher_emb, dim=1)

    # Test that loss is a non-negative scalar
    loss = contrastive_distillation_loss(
        student_embedding_batch=student_emb,
        teacher_embedding_batch=teacher_emb,
        loss_temperature=1.0,
    )
    assert loss.dim() == 0
    assert loss.item() >= 0  # KL divergence is non-negative

    # Test that identical embeddings give zero loss
    identical_loss = contrastive_distillation_loss(
        student_embedding_batch=teacher_emb,
        teacher_embedding_batch=teacher_emb,
        loss_temperature=1.0,
    )
    torch.testing.assert_close(identical_loss, torch.tensor(0.0))


def test_arrow_streaming_shuffle_basic():
    """Test basic streaming shuffle functionality."""
    # Create test data
    schema = pa.schema([("id", pa.int64()), ("value", pa.float32())])
    batch1 = pa.RecordBatch.from_arrays(
        [pa.array([1, 2, 3, 4]), pa.array([1.0, 2.0, 3.0, 4.0])], schema=schema
    )
    batch2 = pa.RecordBatch.from_arrays(
        [pa.array([5, 6, 7, 8]), pa.array([5.0, 6.0, 7.0, 8.0])], schema=schema
    )
    batches = [batch1, batch2]

    def batch_iter() -> Iterator[pa.RecordBatch]:
        for batch in batches:
            yield batch

    # Test shuffling
    results = list(
        arrow_streaming_shuffle(
            record_batch_iterables=[batch_iter(), batch_iter()],
            output_batch_size=3,
            buffer_size=5,
            seed=42,
        )
    )

    # Each input batch is length 4, so 8 examples produces 2 complete batches of 3.
    assert len(results) == 2

    # Each result should be a list of 2 record batches (one from each iterable)
    for batch_pair in results:
        assert len(batch_pair) == 2
        assert isinstance(batch_pair[0], pa.RecordBatch)
        assert isinstance(batch_pair[1], pa.RecordBatch)
        assert len(batch_pair[0]) == 3  # output_batch_size
        assert len(batch_pair[1]) == 3  # output_batch_size

    # Check that the batches are shuffled
    assert not np.array_equal(results[0][0], batches[0])
    assert not np.array_equal(results[0][1], batches[1])
    assert not np.array_equal(results[1][0], batches[1])
    assert not np.array_equal(results[1][1], batches[0])

    # Check that the shuffling is the same for both iterators
    assert np.array_equal(results[0][0], results[0][1])
    assert np.array_equal(results[1][0], results[1][1])


def test_arrow_streaming_shuffle_deterministic():
    """Test that streaming shuffle is deterministic with same seed."""
    schema = pa.schema([("id", pa.int64())])

    def create_batches():
        for i in range(3):
            batch = pa.RecordBatch.from_arrays(
                [pa.array(list(range(i * 4, (i + 1) * 4)))], schema=schema
            )
            yield batch

    def run_shuffle(seed):
        return list(
            arrow_streaming_shuffle(
                record_batch_iterables=[create_batches()],
                output_batch_size=2,
                buffer_size=6,
                seed=seed,
            )
        )

    results1 = run_shuffle(42)
    results2 = run_shuffle(42)
    results3 = run_shuffle(123)  # Different seed

    # Same seed should give same results
    assert len(results1) == len(results2)
    for batch_pair1, batch_pair2 in zip(results1, results2):
        assert batch_pair1[0].equals(batch_pair2[0])

    # Different seed should give different results (with high probability)
    if len(results1) > 0 and len(results3) > 0:
        # At least one batch should be different
        assert not all(
            batch_pair1[0].equals(batch_pair3[0])
            for batch_pair1, batch_pair3 in zip(results1, results3)
        )
