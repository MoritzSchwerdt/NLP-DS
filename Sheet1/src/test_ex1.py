from typing import Final, List
import traceback

tiny_corpus: Final[List[str]] = [
    "After making her film debut in 1984's The Brother from Another Planet, Merediz has appeared in a number of films, including The Milagro Beanfield War (1988), City of Hope (1991), Evita (1996), Music of the Heart (1999), K-Pax (2001),",
    "Helmreich is a surname. Notable people with the surname include:",
    "Wellens also demonstrated that the reproducible initiation and termination of arrhythmias by programmed electrical stimulation of the heart allowed the study of the effect of antiarrhythmic drugs on the mechanism of the arrhythmia. In 1977, he moved to the new University of Limburg in Maastricht, Netherlands, to develop academic cardiology there. He created an internationally known center for the study and treatment of cardiac arrhythmias.",
    "A post office was established at New Era in 1868, and remained in operation until it was discontinued in 1906.",
    'According to the Journal Citation Reports, the journal has a 2019 impact factor of 2.863, ranking it 37h out of 85 journals in the category "Geochemistry & Geophysics".',
    "9th Boucles de l'Aulne",
    "Danil Sergeyevich Lysenko (Russian: Данил Сергеевич Лысенко; born 19 May 1997) is a Russian track and field athlete who specialises in the high jump. He won the silver medal at the 2017 World Championships.",
    "The song was recorded in 2011 with Nianell on their joint project together 'N Duisend Drome. It was released as the Afrikaans version My Engel. In 2012 the song was re-recorded in English and was released as their debut worldwide single.",
    "Kameyama Station (亀山駅, Kameyama-eki) is a junction passenger railway station located in the city of Kameyama, Mie Prefecture, Japan, owned by Central Japan Railway Company (JR Central).",
    '"At The End of The Day"',
]


def test_window():
    from nlpds.submission.ex1 import ContextWindow, TokenizedSentence
    import numpy as np

    tokens = TokenizedSentence(
        [0, 1, 2, 3, 4],
    )
    windows = tokens.get_windows(2, np.ones(5))
    assert windows[0].target == 0, f"expected: {0}, got: {windows[0].target}"
    assert windows[0].context == [
        1,
        2,
    ], f"expected: {[1, 2]}, got: {windows[0].context}"

    assert windows[2].target == 2, f"expected: {0}, got: {windows[2].target}"
    assert windows[2].context == [
        0,
        1,
        3,
        4,
    ], f"expected: {[0, 1, 3, 4]}, got: {windows[2].context}"


def test_tokenizer():
    from nlpds.submission.ex1 import Tokenizer

    tokenizer = Tokenizer()
    tokenizer.train(
        tiny_corpus,
        25,
    )

    # Vocabulary sanity check
    expected_vocabulary = {
        "the": 0,
        ",": 1,
        "of": 2,
        ".": 3,
        "in": 4,
        "(": 5,
        ")": 6,
        "was": 7,
        "The": 8,
    }
    for word, expected_idx in expected_vocabulary.items():
        actual_idx = tokenizer.vocabulary.get(word)
        assert (
            actual_idx == expected_idx
        ), f"Expected '{word}' to have index {expected_idx} but got {actual_idx}"

    return tokenizer


def test_dataset():
    from nlpds.submission.ex1 import Dataset, Tokenizer

    tokenizer = test_tokenizer()

    dataset = Dataset.with_tokenizer(tiny_corpus, tokenizer)

    assert len(dataset) == len(
        tiny_corpus
    ), f"expected dataset length to be equal to tiny_corpus length but got: {len(dataset)} != {len(tiny_corpus)}"

    actual = list(dataset.token_counts)[:5]
    expected = [19, 17, 14, 13, 11]
    assert actual == expected, f"expected: {expected}, got: {actual}"

    assert len(dataset.token_counts) == len(
        dataset.token_frequencies
    ), "word counts and word frequencies should be of equal length"


def test_word2vec():
    import torch
    from nlpds.submission.ex1 import SkipGramSoftMax

    vocab_size = 25
    emb_dim = 10

    model = SkipGramSoftMax(vocab_size, emb_dim)

    emb_target = model.embed_input(2)
    assert emb_target.shape == (1, emb_dim)

    emb_context = model.embed_output(0, 1, 3, 4)
    assert emb_context.shape == (4, emb_dim)

    output_ids = model.forward(2, [0, 1, 3, 4])
    output_tensors = model.forward(emb_target, emb_context)

    assert torch.equal(
        output_ids, output_tensors
    ), "Expected the forward result to be equal for both id and tensor inputs"

    # By setting all weights to 1 ...
    model.set_weights(torch.ones((25, 10)), 2 * torch.ones((25, 10)))

    # ... this should be equal to -3.2189
    assert abs(model.calculate_objective(0, 1).item()) - 3.2189 < 0.0001


VERBOSE: Final[int] = 1

if __name__ == "__main__":
    try:
        test_window()
        print("test_window: passed")
    except:
        print("test_window: failed")
        if VERBOSE:
            traceback.print_exc()

    try:
        test_tokenizer()
        print("test_tokenizer: passed")
    except:
        print("test_tokenizer: failed")
        if VERBOSE:
            traceback.print_exc()

    try:
        test_dataset()
        print("test_dataset: passed")
    except:
        print("test_dataset: failed")
        if VERBOSE:
            traceback.print_exc()
    try:
        test_word2vec()
        print("test_word2vec: passed")
    except:
        print("test_word2vec: failed")
        if VERBOSE:
            traceback.print_exc()

    # Note: There will be further updates of the exercise sheets with more tests!
