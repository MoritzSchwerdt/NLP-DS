import abc
from collections import abc as colletions_abc
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Tuple

import numpy as np
import torch
from torch import Tensor

#######################
# 1.1 Text Processing #
#######################


class ContextWindowABC(abc.ABC):
    # Target input id
    target: int
    # Context input ids
    context: List[int]

    def __len__(self) -> int:
        """Get the actual length of the context.

        Returns:
            int: The total number of tokens in the context window to the left and right.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.__len__() is not implemented"
        )

    def as_tuple(self) -> Tuple[int, List[int]]:
        """Get the target and context as a tuple.

        Returns:
            Tuple[int, List[int]]: A tuple of the target index and a single list containing the left and right context window's token ids, in order.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.as_tuple() is not implemented"
        )


class TokenizedSentenceABC:
    input_ids: List[int]

    def __init__(self, input_ids: List[int]):
        """Construct a new tokenized sentence object from the given input ids.

        Args:
            input_ids (List[int]): The input token ids.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.from_input_ids() is not implemented"
        )

    def get_windows(
        self, window_size: int, subsampling_probabilities: np.ndarray
    ) -> List[ContextWindowABC]:
        """Generate all context windows for this sentence.
        Apply the subsampling rules outlined in the paper according to the given subsampling probabilities.

        Args:
            window_size (int): The window size to the left and right of each target word.
            subsampling_probabilities (np.ndarray): The subsampling probabilities for all tokens in the vocabulary. The values denote the probability to *keep* each token.

        Returns:
            List[ContextWindowABC]: The resulting list of context windows
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.get_windows() is not implemented"
        )


class TokenizerABC(abc.ABC):
    vocabulary: Dict[str, int]

    def __init__(self, lowercase: bool = False):
        """Tokenizer constructor.

        Args:
            lowercase (bool, optional): If True, all casing is ignored during training AND on later use. Defaults to False.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.__init__() is not implemented"
        )

    def train(self, dataset: List[str], max_vocab_size: int):
        """Train a tokenizer on the given dataset.
        The final vocabulary size may not exceed the given maximum size.

        Args:
            dataset (List[str]): The dataset to use for training the tokenizer.
            max_vocab_size (int): The maximum vocabulary size.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.train() is not implemented"
        )

    def _tokens_to_input_ids(self, tokens: List[str]) -> List[int]:
        """Convert a list of token strings to their corresponding input ids.
        All tokens not present in the vocabulary will be removed.

        Args:
            tokens (List[str]): The tokens to convert to their respective ids.

        Returns:
            List[int]: A list of token ids.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}._tokens_to_input_ids() is not implemented"
        )

    def _pre_tokenize(self, sentence: str) -> List[str]:
        """Perform pre-tokenization on the given sentence.

        Args:
            sentence (str): The sentence to pre-tokenize.

        Returns:
            List[str]: The list of token strings.
        """

        raise NotImplementedError(
            f"{self.__class__.__name__}._pre_tokenize() is not implemented"
        )

    def tokenize(self, sentence: str) -> TokenizedSentenceABC:
        """Tokenize the given sentence.

        Args:
            sentence (str): The sentence to tokenize.

        Returns:
            TokenizedSentenceABC: The tokenized sentence.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.tokenize() is not implemented"
        )


class DatasetABC(abc.ABC):
    """
    The dataset object should store the tokenized sentences.
    """

    @classmethod
    def with_tokenizer(cls, corpus: List[str], tokenizer: TokenizerABC):
        """Create a new dataset object from the given raw corpus using the given tokenizer.

        Hint: Count the number of occurrences for each token, you will need it!

        Args:
            corpus (List[str]): The raw input corpus.
            tokenizer (TokenizerABC): The tokenizer to use.
        """
        raise NotImplementedError(f"{cls.__name__}.with_tokenizer() is not implemented")

    def __len__(self) -> int:
        """Get the size of this dataset, i.e. the number of sentences.

        Returns:
            int: The size of this dataset.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.__len__() is not implemented"
        )

    def dataloader(
        self, window_size: int, threshold: float = 0.001, shuffle: bool = False
    ) -> Iterator[List[ContextWindowABC]]:
        """
        Create a dataloader from this dataset, i.e. an Iterator that returns a list of all windows according to given window size for each sentence.
        If shuffle=True is given, the order of sentences should be randomized.

        Hint:
            For background on the requirements for an Iterator, see: https://docs.python.org/3/library/collections.abc.html#collections-abstract-base-classes

        Args:
            window_size (int): The window size to the left and right of the target word.
            threshold (float, optional): The subsampling threshold. Defaults to 0.001.
            shuffle (bool, optional): If True, the order of sentences should be randomized. Defaults to False.

        Yields:
            Iterator[List[ContextWindowABC]]: An iterator of lists containing all ContextWindows for each sentence in the dataset.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.iter() is not implemented"
        )

    @property
    def token_counts(self) -> np.ndarray:
        """Get the raw counts of each token from the given input corpus.
        Should return an array of counts, where each entry (row) corresponds to the token id.

        Returns:
            np.ndarray: A numpy array of integers containing the number of occurrences for each token from the vocabulary of the used tokenizer, according to the tokens' occurrence in the corpus.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.word_counts property getter is not implemented"
        )

    @property
    def token_frequencies(self) -> np.ndarray:
        """Get the frequency of each token in the given input corpus.
        Should return an array of floats, where each entry (row) corresponds to the token id.

        Returns:
            np.ndarray: A numpy array of floats containing the number of occurrences for each token from the vocabulary of the used tokenizer, according to the tokens' occurrence in the corpus.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.word_frequencies property getter is not implemented"
        )

    def subsampling_probability(self, threshold: float = 0.001) -> np.ndarray:
        """Get the probabilities for *keeping* a token.
        Should return an array of floats, where each entry (row) corresponds to the token id.
        All values should be between 0.0 and 1.0.

        Args:
            threshold (float, optional): The subsampling threshold. Defaults to 0.001.

        Returns:
            np.ndarray: A numpy array of floats containing the probability to keep each token from the vocabulary of the used tokenizer, according to the tokens' occurrence count in the corpus.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.subsampling_probability() is not implemented"
        )


##########################
# 1.2 Language Modelling #
##########################


class Word2VecABC(torch.nn.Module, abc.ABC):
    vocab_size: int
    embedding_dim: int
    # Input or target embedding tensor. Must have require_grad=True.
    embedding_input: Tensor
    # Output or context embedding tensor. Must have require_grad=True.
    embedding_output: Tensor

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
    ):
        """The word2vec model base class.

        Args:
            vocab_size (int): The total size of the vocabulary.
            embedding_dim (int): The size of the input and output embeddings.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.__init__() is not implemented"
        )

    def init_weights(self):
        """
        (Re-)Initialize the weights (embeddings).
        Should be called from within Wor2VecSkipGramABC.__init__()
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.init_weights() is not implemented"
        )

    def set_weights(self, weights_input: Tensor, weights_output: Tensor):
        """Set the weights to the given values.

        Args:
            weights_input (Tensor): Input or target embedding values.
            weights_output (Tensor): Output or context embedding values.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.set_weights() is not implemented"
        )

    def embed_input(self, *token_ids: int) -> Tensor:
        """Get the input embeddings of the given tokens.

        Args:
            token_ids (*int): Any number of token ids.

        Returns:
            Tensor: A single Tensor of shape (len(token_ids), self.embedding_dim).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.embed_input() is not implemented"
        )

    def embed_output(self, *token_ids: int) -> Tensor:
        """Get the output embeddings of the given tokens.

        Args:
            token_ids (*int): Any number of token ids.

        Returns:
            Tensor: A single Tensor of shape (len(token_ids), self.embedding_dim).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.embed_output() is not implemented"
        )

    def forward(self, target: int | Tensor, context: List[int] | Tensor) -> Tensor:
        """Perform a forward pass, i.e. calculate the dot-product of the embedding of the given target token with each embedding of the given context tokens
        Args:
            target (int | Tensor): The target token id or a 1D embedding tensor.
            context (List[int] | Tensor): Any number of context token ids or their embeddings in a 2D Tensor.

        Hint:
            Either pass the raw token ids to this method or embed the tokens first, e.g. using the appropriate embed_*() method.

        Returns:
            Tensor: Returns a 1D Tensor of size c, where c is the number of given context tokens, containing the dot-products of the target embedding with the context embeddings.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.forward() is not implemented"
        )

    def calculate_objective(self, *args, **kwargs) -> Tensor:
        """Calculate the objective for the given sample.

        Note:
            You do not need to use this method in your training loop, if there is a more efficient way to obtain the loss.
            You must, however, implement this method for evaluation purposes.

        Returns:
            Tensor: The SoftMax loss of the given sample.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.calculate_objective() is not implemented"
        )
