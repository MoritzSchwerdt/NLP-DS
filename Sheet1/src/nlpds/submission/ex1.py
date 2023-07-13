import this
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterator
import re

import torch
from torch import Tensor

import numpy as np

from nlpds.abc.ex1 import (
    DatasetABC,
    ContextWindowABC,
    TokenizedSentenceABC,
    TokenizerABC,
    Word2VecABC,
)

#######################
# 1.1 Text Processing #
#######################


@dataclass
class ContextWindow(ContextWindowABC):
    """Example: This is how you may implement the interfaces."""

    # Target input id
    target: int
    # Context input ids
    context: List[int]

    def __len__(self) -> int:
        """Get the actual length of the context.

        Returns:
            int: The total number of tokens in the context window to the left and right.
        """
        return len(self.context)

    def as_tuple(self) -> Tuple[int, List[int]]:
        """Get the target and context as a tuple.

        Returns:
            Tuple[int, List[int]]: A tuple of the target index and a single list containing the left and right context window's token ids, in order.
        """
        return self.target, self.context


class TokenizedSentence(TokenizedSentenceABC):
    input_ids: List[int]

    def __init__(self, input_ids: List[int]):
        """Construct a new tokenized sentence object from the given input ids.

                Args:
                    input_ids (List[int]): The input token ids.
                """
        self.input_ids = input_ids

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
        context_windows = []
        #iterate ids
        for i in range(len(self.input_ids)):
            #check subsampling
            if np.random.rand() < subsampling_probabilities[self.input_ids[i]]:
                #allow first and last indexes as input word
                start = max(0, i - window_size)
                end = min(len(self.input_ids), i + window_size + 1)
                #create single context_window to add
                context = self.input_ids[start:i] + self.input_ids[i + 1:end]
                context_windows.append(ContextWindow(self.input_ids[i], context))
        return context_windows
class Tokenizer(TokenizerABC):
    lowercase: bool
    vocabulary: Dict[str,str]
    def __init__(self, lowercase: bool = False):
        """Tokenizer constructor.

        Args:
            lowercase (bool, optional): If True, all casing is ignored during training AND on later use. Defaults to False.
        """
        self.lowercase = lowercase

    def train(self, dataset: List[str], max_vocab_size: int):
        """Train a tokenizer on the given dataset.
        The final vocabulary size may not exceed the given maximum size.

        Args:
            dataset (List[str]): The dataset to use for training the tokenizer.
            max_vocab_size (int): The maximum vocabulary size.
        """
        #tokenize sentences
        tokens = [self._pre_tokenize(sentence) for sentence in dataset]
        tokens = np.concatenate(tokens).tolist()

        #get frequency distributions of tokens
        unique_tokens, counts = np.unique(tokens, return_counts=True)
        token_freq_dist = dict(zip(unique_tokens, counts))

        most_common_tokens = sorted(token_freq_dist, key=token_freq_dist.get, reverse=True)[:max_vocab_size]
        self.vocabulary = {token: i for i, token in enumerate(most_common_tokens)}

    def _tokens_to_input_ids(self, tokens: List[str]) -> List[int]:
        """Convert a list of token strings to their corresponding input ids.
        All tokens not present in the vocabulary will be removed.

        Args:
            tokens (List[str]): The tokens to convert to their respective ids.

        Returns:
            List[int]: A list of token ids.
        """
        input_ids = [self.vocabulary[token] for token in tokens if token in self.vocabulary]
        return input_ids

    def _pre_tokenize(self, sentence: str) -> List[str]:
        """Perform pre-tokenization on the given sentence.

        Args:
            sentence (str): The sentence to pre-tokenize.

        Returns:
            List[str]: The list of token strings.
        """
        sentence = sentence.lower() if self.lowercase else sentence
        return re.findall(r'\b\w+\b|\S', sentence)

    def tokenize(self, sentence: str) -> TokenizedSentenceABC:
        """Tokenize the given sentence.

        Args:
            sentence (str): The sentence to tokenize.

        Returns:
            TokenizedSentenceABC: The tokenized sentence.
        """
        tokens = self._pre_tokenize(sentence)
        input_ids = self._tokens_to_input_ids(tokens)
        return TokenizedSentence(input_ids)

class Dataset(DatasetABC):
    """
        The dataset object should store the tokenized sentences.
        """
    tokenizer: TokenizerABC
    corpus: List[str]

    def __init__(self, corpus: List[str], tokenizer: TokenizerABC):
        self.corpus = corpus
        self.tokenizer = tokenizer

    @classmethod
    def with_tokenizer(cls, corpus: List[str], tokenizer: TokenizerABC):
        """Create a new dataset object from the given raw corpus using the given tokenizer.

        Hint: Count the number of occurrences for each token, you will need it!

        Args:
            corpus (List[str]): The raw input corpus.
            tokenizer (TokenizerABC): The tokenizer to use.
        """
        return cls(corpus, tokenizer)

    def __len__(self) -> int:
        """Get the size of this dataset, i.e. the number of sentences.

        Returns:
            int: The size of this dataset.
        """
        return len(self.corpus)

    def dataloader(
            self, window_size: int, threshold: float = 0.001, shuffle: bool = False
    ) -> Iterator[List[ContextWindow]]:
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
        sentences = self.corpus

        if(shuffle):
            np.random.shuffle(sentences)

        #iterate sentences to tokenize
        for sentence in sentences:
            tokenized_sentence = self.tokenizer.tokenize(sentence)
            yield tokenized_sentence.get_windows(window_size, self.subsampling_probability(threshold))

    @property
    def token_counts(self) -> np.ndarray:
        """Get the raw counts of each token from the given input corpus.
        Should return an array of counts, where each entry (row) corresponds to the token id.

        Returns:
            np.ndarray: A numpy array of integers containing the number of occurrences for each token from the vocabulary of the used tokenizer, according to the tokens' occurrence in the corpus.
        """
        token_counts = [0] * len(self.tokenizer.vocabulary)

        for sentence in self.corpus:
            tokenized_sentence = self.tokenizer.tokenize(sentence)
            for token_id in tokenized_sentence.input_ids: token_counts[token_id] += 1

        return np.array(token_counts)

    @property
    def token_frequencies(self) -> np.ndarray:
        """Get the frequency of each token in the given input corpus.
        Should return an array of floats, where each entry (row) corresponds to the token id.

        Returns:
            np.ndarray: A numpy array of floats containing the number of occurrences for each token from the vocabulary of the used tokenizer, according to the tokens' occurrence in the corpus.
        """
        total_amount_tokens = np.sum(self.token_counts)
        return self.token_counts / total_amount_tokens

    def subsampling_probability(self, threshold: float = 0.001) -> np.ndarray:
        """Get the probabilities for *keeping* a token.
        Should return an array of floats, where each entry (row) corresponds to the token id.
        All values should be between 0.0 and 1.0.

        Args:
            threshold (float, optional): The subsampling threshold. Defaults to 0.001.

        Returns:
            np.ndarray: A numpy array of floats containing the probability to keep each token from the vocabulary of the used tokenizer, according to the tokens' occurrence count in the corpus.
        """
        frequencies = self.token_frequencies

        probabilities_keeping = (np.sqrt(frequencies / threshold) + 1) * (threshold / frequencies)

        return probabilities_keeping

##########################
# 1.2 Language Modelling #
##########################


def onehot_vector(indices: List[int], size: int) -> Tensor:  # TODO
    """Returns a 2D one-hot encoded Tensor of the given size for the given indices.

    Args:
        indices (List[int]): A list of token indices.
        size (int): The size of the one-hot vectors.

    Returns:
        Tensor: A 2D tensor of shape (len(indices), size).
    """
    one_hot = torch.zeros(len(indices), size)
    one_hot[torch.arange(len(indices)), indices] = 1
    return one_hot


# Hint: It may be useful to introduce another class like 'Word2VecBase'
# that inherits from Word2VecABC instead to remain true to the DRY principle.
#   - DRY => Don't Repeat Yourself
#
# While not required, it could look like this:
#
# class Word2VecBase(Word2VecABC):
#     pass
# class SkipGramSoftMax(Word2VecBase):
#     pass
# class SkipGramNegativeSampling(Word2VecBase):
#     pass

class Word2VecBase(Word2VecABC):
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
        torch.nn.Module.__init__(self)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_input = torch.nn.Embedding(vocab_size, embedding_dim)
        self.embedding_output = torch.nn.Embedding(vocab_size, embedding_dim)
        self.init_weights()

    def init_weights(self):
        """
        (Re-)Initialize the weights (embeddings).
        Should be called from within Wor2VecSkipGramABC.__init__()
        """
        initrange = 0.5 / self.embedding_dim
        self.embedding_input.weight.data.uniform_(-initrange, initrange)
        self.embedding_output.weight.data.uniform_(-0, 0)

    def set_weights(self, weights_input: Tensor, weights_output: Tensor):
        """Set the weights to the given values.

        Args:
            weights_input (Tensor): Input or target embedding values.
            weights_output (Tensor): Output or context embedding values.
        """
        self.embedding_input.weight.data = weights_input
        self.embedding_output.weight.data = weights_output

    def embed_input(self, *token_ids: int) -> Tensor:
        """Get the input embeddings of the given tokens.

        Args:
            token_ids (*int): Any number of token ids.

        Returns:
            Tensor: A single Tensor of shape (len(token_ids), self.embedding_dim).
        """
        token_id_tensor = torch.tensor(token_ids, dtype=torch.long)
        return self.embedding_input(token_id_tensor)

    def embed_output(self, *token_ids: int) -> Tensor:
        """Get the output embeddings of the given tokens.

        Args:
            token_ids (*int): Any number of token ids.

        Returns:
            Tensor: A single Tensor of shape (len(token_ids), self.embedding_dim).
        """
        token_id_tensor = torch.tensor(token_ids, dtype=torch.long)
        return self.embedding_output(token_id_tensor)

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
        #if target / context are tensors, they are embeddings already
        embed_target = Tensor()
        embed_context = Tensor()
        if isinstance(target, Tensor):
            this.embed_target = target
        else:
            this.embed_target = self.embed_input(target)

        if isinstance(context, Tensor):
            this.embed_context = context
        else:
            this.embed_context = self.embed_output(context)

        #get scores of target and context
        scores = torch.sum(embed_target * embed_context, dim=-1)
        return scores

    #left to implement in ChildrenClasses (DRY)
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


class SkipGramSoftMax(Word2VecBase):

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__(vocab_size, embedding_dim)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
    def calculate_objective(self, target_id: int, context_id: int) -> Tensor:
        """Calculate the SoftMax objective according to equations (1) & (2) in the word2vec paper (Mikolov, 2013, NIPS).

        Note:
            You do not need to use this method in your training loop.
            You must, however, implement this method for evaluation purposes.

        Returns:
            Tensor: The SoftMax objective for the given sample.
        """
        #get embeddings
        target_embedding = self.embed_input(target_id)
        context_embedding = self.embed_output(context_id)

        #calculate probs according to paper
        logits = torch.matmul(target_embedding, context_embedding.t())
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        objective = log_probs[0:context_id]

        return objective


class SkipGramNegativeSampling(Word2VecBase):

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__(vocab_size, embedding_dim)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

    def calculate_objective(
        self, target_id: int, context_id: int, negative_sample_ids: List[int]
    ) -> Tensor:
        """Calculate the Negative Sampling objective according to equation (4) in the word2vec paper (Mikolov, 2013, NIPS).

        Note:
            You do not need to use this method in your training loop.
            You must, however, implement this method for evaluation purposes.

        Returns:
            Tensor: The Negative Sampling objective for the given sample.
        """
        #get embeddings
        target_embedding = self.embed_input(target_id)
        context_embedding = self.embed_output(context_id)
        negative_samples_embedding = self.embed_output(negative_sample_ids)

        positive_score = torch.dot(target_embedding, context_embedding)
        negative_score = torch.sum(torch.dot(target_embedding, negative_samples_embedding))

        #calc objective
        objective = -torch.log(torch.sigmoid(positive_score)) - torch.log(torch.sigmoid(-negative_score))

        return objective


##################
# Training Loops #
##################


def train_sf(
    model: SkipGramSoftMax,
    dataset: Dataset,
    epochs: int,
    learning_rate: float,
    threshold: float,
):
    """Run word2vec training using SoftMax on the given dataset for the given number of epochs (called 'iterations' in the paper) with the given learning rate.

    Args:
        model (SkipGramSoftMax): The word2vec model.
        dataset (Dataset): The dataset to use for training.
        epochs (int): The number of training iterations.
        threshold (float): The sub-sampling threshold to use.
        learning_rate (float): The learning rate to use.
    """
    raise NotImplementedError(f"train_sf() is not implemented")


def train_ns(
    model: SkipGramNegativeSampling,
    dataset: Dataset,
    epochs: int,
    learning_rate: float,
    threshold: float,
    k: int,
):
    """Run word2vec training using Negative Sampling with k given negative samples on the given dataset for the given number of epochs (called 'iterations' in the paper) with the given learning rate.

    Args:
        model (SkipGramNegativeSampling): The word2vec model.
        dataset (Dataset): The dataset to use for training.
        epochs (int): The number of training iterations.
        learning_rate (float): The learning rate to use.
        threshold (float): The sub-sampling threshold to use.
        k (float): The number of negative samples to use.
    """
    raise NotImplementedError(f"train_ns() is not implemented")


###############
# Bonus: CBOW #
###############


class CbowSoftMax(Word2VecABC):
    pass  # TODO


###################
# Bonus: wang2vec #
###################


class StructuredSkipGramSoftMax(SkipGramSoftMax):
    pass  # TODO


class StructuredSkipGramNegativeSampling(SkipGramNegativeSampling):
    pass  # TODO
