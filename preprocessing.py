from byte_pair_encoding import Tokenizer, bpe_generator
from language_model_dataset import truncate_batch
import torch
from torch.utils.data import DataLoader
import pickle
import os
from random import shuffle
from typing import Generator
from gutenberg_preprocess import gutenberg_paragraphs


def create_corpus(corpus_dir: str) -> Generator[str, None, None]:
    for root, _, files in os.walk(corpus_dir):
        for file in files:
            print(file)
            file = os.path.join(corpus_dir, file)
            try:
                with open(file, "r", encoding="utf-8") as f:
                    yield f.read()
            except Exception as e:
                print(e)
                pass


def create_lm_dataloaders() -> tuple[dict[str, DataLoader], [str]]:
    max_texts = 10000
    with open("datasets/gutenberg_tokens.txt", "r", encoding="utf-8") as f:
        list_str = f.read()
    tokens = eval(list_str)
    tokenizer = Tokenizer(tokens)
    if os.path.exists("datasets/lm_gutenberg_dataset_train.pkl"):
        with open("datasets/lm_gutenberg_dataset_train.pkl", "rb") as f:
            lm_dataset_train = pickle.load(f)
        with open("datasets/lm_gutenberg_dataset_test.pkl", "rb") as f:
            lm_dataset_test = pickle.load(f)
    else:
        corpus = list(gutenberg_paragraphs())
        shuffle(corpus)
        token_corpus = []
        for idx, text in enumerate(corpus[:max_texts]):
            if idx % 100 == 0:
                print(f"{idx}/{min(max_texts, len(corpus))}")
            try:
                token_text = list(tokenizer.tokenize(text))
                token_corpus.append(token_text)
            except ValueError:
                pass
        print()
        shuffle(token_corpus)
        test_cutoff = 400
        lm_dataset_test = [torch.tensor(text, dtype=torch.long) for text in token_corpus[:test_cutoff]]
        lm_dataset_train = [torch.tensor(text, dtype=torch.long) for text in token_corpus[test_cutoff:]]
        with open("datasets/lm_gutenberg_dataset_test.pkl", "wb") as f:
            pickle.dump(lm_dataset_test, f)
        with open("datasets/lm_gutenberg_dataset_train.pkl", "wb") as f:
            pickle.dump(lm_dataset_train, f)
    data_loaders = {"train": DataLoader([text for text in lm_dataset_train if text.numel()],
                                        batch_size=100, shuffle=True,
                                        num_workers=1, collate_fn=truncate_batch),
                    "test":  DataLoader([text for text in lm_dataset_test if text.numel()],
                                        batch_size=600, shuffle=False,
                                        num_workers=1, collate_fn=truncate_batch)
                    }
    return data_loaders, tokens


def bpe_for_gutenberg():
    bpe_generator(list(gutenberg_paragraphs()), 1024)

