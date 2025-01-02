import numpy as np
import torch
import os
from torch.utils.data import Dataset
from bisect import bisect_left


def truncate_batch(batch: [torch.Tensor]) -> torch.Tensor:
    batch_data_sizes = [len(x) for x in batch]
    min_data_size = min(batch_data_sizes)
    for idx, datum in enumerate(batch):
        batch[idx] = datum[:min_data_size]
    data = torch.stack(batch)
    return data


class LanguageModelDataSet(Dataset):
    def __init__(self, corpus: [[int]]):
        self.corpus = corpus
        self.indexing_table = [0]

        for text in self.corpus[:-1]:
            # - 1 is there because a whole text isn't a valid training example.
            # There must be another token to be predicted
            self.indexing_table.append(self.indexing_table[-1] + len(text) - 1)

    def __len__(self):
        return sum([len(text)-1 for text in self.corpus])

    def __getitem__(self, idx):
        # idx is the index of the target
        text_index = bisect_left(self.indexing_table, idx + 0.5)
        text = self.corpus[text_index-1]
        # The 0th token can can't be a target because of model architecture
        target_index = idx+1 - self.indexing_table[text_index-1]
        data = torch.tensor(text[:target_index], dtype=torch.long)
        target = torch.tensor(text[target_index], dtype=torch.long)
        return data, target
