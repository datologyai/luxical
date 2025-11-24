import pyarrow as pa

__version__: str

class ArrowTokenizer:
    """A tokenizer that operates on pyarrow arrays, implemented in Rust."""

    def __init__(self, json_content: str) -> None:
        """
        Initializes the tokenizer from the JSON content of a tokenizer file.

        Args:
            json_content: The string content of the tokenizer's JSON file.
        """
        ...

    def to_str(self) -> str:
        """
        Serializes the tokenizer to a JSON string.

        Returns:
            A string containing the tokenizer's configuration as JSON.
        """
        ...

    def tokenize(
        self, texts: pa.StringArray, add_special_tokens: bool = False
    ) -> pa.LargeListArray:
        """
        Tokenizes a pyarrow StringArray of texts.

        Args:
            texts: A pyarrow.StringArray containing the texts to tokenize.
            add_special_tokens: Whether to add special tokens. Defaults to False.

        Returns:
            A pyarrow.LargeListArray where each item is a list of token IDs (as uint32).
        """
        ...
