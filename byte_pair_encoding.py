import os
from typing import Generator
import re


def count_and_replace_pairs(token_corpus: [[int]], new_token_id: int | None,
                            pair_to_be_replaced: tuple[int, int] | None) -> tuple[dict[tuple[int, int], int], [int]]:
    pair_counts = {}
    for text_index, text in enumerate(token_corpus):
        for index in range(len(text) - 1):
            prev_token = text[index]
            token = text[index + 1]
            if prev_token is None or token is None:
                continue
            pair = prev_token, token
            if pair == pair_to_be_replaced:
                if index > 0:
                    prev_pair = text[index - 1], text[index]
                    if prev_pair in pair_counts:
                        pair_counts[prev_pair] -= 1

                    new_prev_pair = text[index - 1], new_token_id
                    if new_prev_pair in pair_counts:
                        pair_counts[new_prev_pair] += 1
                    else:
                        pair_counts[new_prev_pair] = 1

                text[index] = new_token_id
                text[index + 1] = None
                if index + 2 >= len(text):
                    continue
                pair = new_token_id, text[index + 2]

            if pair in pair_counts:
                pair_counts[pair] += 1
            else:
                pair_counts[pair] = 1

        token_corpus[text_index] = [token for token in text if token is not None]
    return pair_counts, token_corpus


def bpe_generator(corpus: [str], target_token_number) -> [str]:
    tokens = []
    token_dict = {}
    token_corpus = []
    for text in corpus:
        for c in text:
            if c not in tokens:
                tokens.append(c)
                token_dict[c] = len(tokens) - 1
        token_corpus.append([token_dict[c] for c in text])
    new_token_id = None
    pair_to_be_replaced = None
    while len(tokens) < target_token_number:
        print(len(tokens))
        pair_counts, token_corpus = count_and_replace_pairs(token_corpus, new_token_id, pair_to_be_replaced)
        for pair in sorted(pair_counts, key=pair_counts.get, reverse=True):
            new_token = tokens[pair[0]] + tokens[pair[1]]
            if any(char.isspace() for char in new_token.strip()) or '\n' in new_token:
                continue
            pair_to_be_replaced = pair
            break
        tokens.append(new_token)
        new_token_id = len(tokens) - 1
        token_dict[new_token] = new_token_id
        print(list(reversed(tokens)))

    return tokens


class Tokenizer:
    def __init__(self, tokens: [str]):
        self.tokens = sorted(tokens, key=len, reverse=True)
        self.token_dict = {token: token_id for token_id, token in enumerate(self.tokens)}
        self.regex_pattern = '|'.join(map(re.escape, self.tokens))
        self.compiled_regex = re.compile(self.regex_pattern)

    def next_token(self, text: str) -> tuple[int, str]:
        match = re.match(self.compiled_regex, text)
        if match:
            return self.token_dict[match.group()], text[match.end():]
        else:
            raise ValueError("Text can't be represented by any token")

    def tokenize(self, text: str) -> Generator[int, None, None]:
        while text:
            token, text = self.next_token(text)
            yield token

    def token_to_str(self, token: int) -> str:
        return self.tokens[token]

    def detokenize(self, tokens: [int]) -> Generator[str, None, None]:
        for token in tokens:
            yield self.token_to_str(token)
