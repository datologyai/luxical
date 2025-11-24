# Luxical — lexical dense embeddings

Fast lexical embeddings built on token counts, TF–IDF, and compact sparse‑to‑dense MLPs.

> :fast_forward: For the ***Quickest*** Start, check out our [Luxical One model on HuggingFace](https://huggingface.co/DatologyAI/luxical-one).

> :warning: **NOTE:** No Planned Active Maintainance :warning:
> 
> This GitHub repository is made available for reproducibility and the advancement of scientific research into fast text embedding methods. At DatologyAI we are proudly comitted to our customers, which unfortunately limits the time we have to actively monitor this repository, accept PRs, or otherwise maintain this project.

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture Notes](#architecture-notes)
- [Training](#training)
- [Development](#development)
- [Release Notes](#release-notes)

## Quick Start

### Installation

```shell
pip install luxical
```

We currently support MacOS and Linux and Python versions 3.11, 3.12, and 3.13. These limitations are due to the inclusion of complied Rust extension code via the included `arrow-tokenize` package.

### Using HuggingFace `transformers`

We have a basic HuggingFace integration that supports inference only. Use of this packaging requires 

``` python
from transformers import AutoModel
from pathlib import Path

# Load from the Huggingface Hub (public or you are logged in for private repos)
hub_model = AutoModel.from_pretrained(
    "datologyai/luxical-one",
    trust_remote_code=True,
)

# Load from a local export directory
local_dir = Path("~/Downloads/luxical_one_hf").expanduser()
local_model = AutoModel.from_pretrained(local_dir, trust_remote_code=True)

emb = local_model(["Luxical integrates with Huggingface."]).embeddings
```

### Using the native `luxical` package APIs

``` python
from pathlib import Path

import luxical.embedder
import luxical.misc_utils

local_dir = Path("~/Downloads/luxical_one.npz").expanduser()
embedder = luxical.embedder.Embedder.load(str(model_local_path))
emb = embedder(["Luxical goes", "very fast"], progress_bars=True)


# EXTRA
# Luxical models typically experience no quality degradation from uint8 quantization.
# Additionally, file-formats supporting dictionary-encoding-based compression
# (e.g. Parquet) may automatically compress roundtrip-quantized data by 4x.
emb_uint8 = luxical.misc_utils.fast_8bit_uniform_scalar_quantize(emb, limit=0.5)
emb_roundtrip = luxical.misc_utils.dequantize_8bit_uniform_scalar_quantized(
    emb_quantized, limit=0.5
)

# EXTRA
# Luxical ships with helper methods to integrate with pyarrow.
import pyarrow as pa
import pyarrow.parquet as pq

emb_pyarrow = luxical.misc_utils.numpy_ndarray_to_pyarrow_fixed_size_list_array(
    emb_roundtrip
)
emb_table = pa.table(
    {
        "document_id": ["doc_1", "doc_2"],
        "embedding": emb_pyarrow,
    }
)
pq.write_table(emb_table, "fast_and_small_embeddings.parquet")
```

## Architecture Notes

### Lexical + Dense
Luxical uses *lexical* (word-based) features. To do this, it tokenizes input text and constructs a Term Frequency (TF) representation over the tokens (aka a "bag of words" featurization). It then applies an Inverse Document Frequency (IDF) scaling and L2 normalization to produce a sparse unit vector.

Luxical is not fully lexical, though. After constructing the sparse unit vector of TF-IDF features, a small feed-forward ReLU neural network maps these features to a dense, normalized embedding.

### Sparse‑by‑dense Is Fast
Consider a sparse feature vector and a dense weight matrix. Multiplying them involves only the columns corresponding to nonzero entries in the sparse vector:

$$
\begin{aligned}
\mathbf{s} &= \begin{bmatrix} 0 & s_2 & 0 & 0 & s_5 & 0 \end{bmatrix}^{\intercal} \\
A &= \begin{bmatrix} \mathbf{a_1} & \mathbf{a_2} & \mathbf{a_3} & \mathbf{a_4} & \mathbf{a_5} & \mathbf{a_6} \end{bmatrix} \\
A\mathbf{s} &= s_1\,\mathbf{a_1} + s_2\,\mathbf{a_2} + s_3\,\mathbf{a_3} + s_4\,\mathbf{a_4} + s_5\,\mathbf{a_5} + s_6\,\mathbf{a_6} \\
&= s_2\,\mathbf{a_2} + s_5\,\mathbf{a_5}
\end{aligned}
$$

Specifically, only the columns in $A$ corresponding to nonzero entries in $\mathbf{s}$ contribute to the operation. This means an optimized implementation of sparse‑by‑dense matmul can run very fast even without GPU acceleration, making Luxical run quite fast.

### A Focus On Efficiency
- Efficient sparse‑by‑dense matmuls (written with Numba) achive high performance on CPU in the sparse-to-dense projection.
- Custom efficient IDF-scaling code avoids performance penalty from IDF-scaling
    - Another approach to speeding up this step would be fusing the IDF-scaling weights directly into the model weights. If $A$ is the dense-to-sparse projection matrix, $\mathbf{b}$ is the IDF weight vector, and $\mathbf{x}$ is the sparse bag-of-words vector, we can IDF-scale the projection matrix $A$, i.e. matmul ordering $(A\,\mathrm{diag}(\mathbf{b}))\,\mathbf{x}$ instead of IDF-scaling the input, i.e. matmul ordering $A\,(\mathrm{diag}(\mathbf{b})\,\mathbf{x})$. Although this lets us scale the weights once ahead of time for all inputs, this fusion complexifies the implementation of training because it changes the parameterization of the model and thus affects the trajectory of gradient-based optimization. Our fast scaling code sidesteps this issue.
- Shallow MLP with ReLU and normalization between layers provides more representation power basically for free.
    - A shallow network with nonlinearity provides noticeably better embedding quality at modest compute. The additional FLOPs cost is small compared with tokenization overhead, so our small MLP addition retains CPU‑friendly performance while improving accuracy over a single linear projection.

### Training Via Distillation
Luxical student embeddings are trained to match pairwise similarities from a teacher model using a KL‑divergence on temperature‑scaled Gram matrices.

## Training

Luxical uses a simple distillation objective to match a teacher model's pairwise similarities.

Core ideas:
- Compute Gram matrices for student and teacher mini‑batches.
- Remove the self‑similarity diagonal to avoid trivial peaks.
- Apply temperature scaling and compute KL divergence (`log_target=True`).

Relevant functions: `remove_diagonal`, `contrastive_distillation_loss` in `src/luxical/training.py`.

Pseudo‑code:

```python
tau = 3.0  # Set the temperature.
S = normalize(student(X))        # [B, D]
T = normalize(teacher(X))        # [B, D]
G_s = S @ S.T
G_t = T @ T.T
G_s = remove_diagonal(G_s)
G_t = remove_diagonal(G_t)
loss = tau**2 * KLDiv(log_softmax(G_s/tau), log_softmax(G_t/tau))
```

See [`./projects/luxical_one`](./projects/luxical_one/) for example training code that walks through the steps of:
1. Embedding a training corpus with a teacher model.
2. Identifying a body of common ngrams from a training corpus and determining approximate inverse-document-frequency scaling for these terms.
3. Training the core embedder parameters via knowledge distillation.


## Development

### Setup

- Install `just` and set up the dev environment:
  - macOS: `brew install just`
  - Ubuntu: `sudo apt-get install -y just`
- For Rust builds (`arrow_tokenize`):
  - macOS: install the Rust toolchain locally.
  - Linux wheels: Docker is required for manylinux targets.
- Helpful commands:
  - `just help` — list tasks
  - `just setup-dev` — create the Python env for development
  - `just lint` — autoformat, lint (with autofix), and typecheck
  - `just test` — run tests

### Building and Publishing

For the core Luxical codebase (in `src`):
- Local wheel: `just build-luxical`
- Publish to the configured index: `just publish-luxical --no-dry-run`

For the `arrow-tokenize` Rust extension:
- macOS local wheels: `just build-arrow-tokenize-macos-local`
- Linux wheels via Docker: `just build-arrow-tokenize-linux-cross`
- Publish: `just publish-arrow-tokenize --no-dry-run`

### Versioning
Version sources are in code, and builds read from those sources when creating wheels.

- luxical
  - Source of truth: `src/luxical/__about__.py:1` → `__version__ = "X.Y.Z"`
  - Runtime API: `from luxical import __version__`

- arrow_tokenize (Rust extension)
  - Source of truth: `arrow_tokenize/Cargo.toml` → `[package].version = "A.B.C"`
  - Runtime API: `import arrow_tokenize as at; at.__version__`

### Cutting a Release
1) Choose patch/minor/major version. Versions may be separate for `luxical` and `arrow-tokenize`.
2) Bump version(s):
   - for luxical: edit `src/luxical/__about__.py`.
   - for arrow_tokenize: edit `arrow_tokenize/Cargo.toml`.
3) Update `README.md` with a dated entry for Luxical updates. Update the Release Notes section of `arrow_tokenize/README.md` for arrow_tokenize updates.
4) Publish to PyPI (see below)
5) Optional: tag versions and push tags, e.g. `git tag vX.Y.Z` or `git tag arrow-tokenize-vA.B.C` followed by `git push --tags`.

### Pushing to PyPI

Get an API token from PyPI (under the [account settings](https://pypi.org/manage/account/) page). Add it to your `~/.pypirc`

```toml
# Contents of ~/.pypirc
[testpypi]
  repository = https://upload.pypi.org/legacy/
  username = __token__
  password = pypi-<something>
[pypi]
  repository = https://test.pypi.org/legacy/
  username = __token__
  password = pypi-<something>
```

``` shell
just publish-wheel-luxical
just publish-wheel-arrow-tokenize
```

## Release Notes

### v1.0.0 — 2025-09-22
- Initialize release notes and publishing workflow.
