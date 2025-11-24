# Luxical One: Distilling Arctic 2M Embeddings On Fineweb Docs

Luxical One is the first non-prototype Luxical embedding model. It distills Arctic 2.0 M into a small lexical-feature-based embedding model. 

## Training Data

The training data consists of a subsample of Fineweb consisting of about 50 million documents (~35 billion tokens). The data was saved to parquet format with `id` and `text` columns. In step 1, we embed all of these documents using the [Arctic Embed 2.0 M](huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0) model. To minimize storage consumption, we use 256-dimensional MRL truncation and uint8 quantization.

## Tokenization And Ngrams

Luxical One adopts the `google-bert/bert-base-uncased` tokenizer, which follows some conventions (e.g. of case-insensitivity) that are commonly used in lexical workflows.

As step 2, we apply an implementation of the Space-Saving Summary algorithm (see [1]) to approximately detect and count the most frequent 5-grams out of a 5M document subsample of the training data. In step 3 (training), we set the vocabulary and inverse document frequency scaling based upon these summary statistics.

[1]: [https://www.cs.ucsb.edu/sites/default/files/documents/2005-23.pdf](https://www.cs.ucsb.edu/sites/default/files/documents/2005-23.pdf)

## Training

In step 3, we train the embedder portion of the Luxical model for three epochs through our training data. We use a large batch size and high loss temperature scaling. We use the Adam optimizer and a  warmup-stable-decay learning rate schedule.
## Evaluation

The core evaluation is performed in `eval_on_fineweb_edu.ipynb` as a reproduction of the Fineweb EDU classifier (i.e. fitting the [HuggingFaceFW/fineweb-edu-llama3-annotations](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-llama3-annotations) supervised classification dataset). For simplicity, we binarize the 5-class classification objective (scores of 1, 2, and 3 are negative, 4 and 5 are positive). Since only the training data annotations were released, we split off our own stratified 10% hold-out test set for this evaluation.

As a baseline, we train a well-tuned FastText classifier on this data. The baseline achieves an AUC score of 0.939 and performs inference on the 40k test documents in 5-10 seconds on a high-end laptop.

To compare Luxical One to this baseline, we embed the training and test data. Luxical One achieves embedding speeds of around 6.5k documents per second, matching the speed of the FastText classifier on raw throughput.

To evaluate the quality of the embeddings, we first round-trip the embedding vectors through 8-bit quantization to demonstrate that this compression is compatible with high accuracy. We then train two scikit-learn classifiers, a logistic regression model and a small 2-layer MLP neural network, on the embedded training examples. The logistic regression model performs inference on 40k documents in around 2ms (meaning throughput is in the tens of millions of input vectors per second) and scores a 0.939 AUC score. The MLP clocks in a bit slower at around 100ms (still a throughput of several hundred thousand documents per second on just a CPU) and achieves a higher AUC score of 0.952.

Given the comparable speed and potential for improved quality, Luxical One represents a compelling tool for fast and accurate "is-edu" classification alone, even before we consider the possibilities unlocked by re-using its embeddings for additional tiny classification models or other tasks like clustering/retrieval.

## Ideas for Luxical Two

- Move to a multilingual corpus with math and code and distill a multilingual teacher model that also handles math and code
- Experiment with different tokenizations approaches and different ngram configurations
- Experiment with even smaller embedding sizes
- Experiment with a deeper model, residual connections, and other architectural variations
- Try skip-grams or other fast lexical featurization methods
