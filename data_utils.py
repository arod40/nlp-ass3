import csv
import linecache
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

NUMBER_TOKENS = 3536


class CharacterLanguageModelDataset(Dataset):
    def __init__(self, data_csv, padded=True):
        self._data_csv = data_csv
        self._total_data = 0
        self._padded = padded
        print("Reading data...")
        with open(self._data_csv, "r") as f:
            for line in tqdm(f.readlines()):
                self._total_data += 1
        self._total_data -= 2  # removing header line and last empty line
        print("DONE!")

    def __getitem__(self, idx):
        id, tokens = linecache.getline(self._data_csv, idx + 2).split(",")

        tokens = [int(t) for t in tokens.split() if self._padded or t != "0"]

        # This is for the bug in the token, it shouldn't be here!
        tokens = [(t if t <= 3536 else 0) for t in tokens]
        X, y = torch.LongTensor(tokens[:-1]), tokens[-1]
        return id, (X, y)

    def __len__(self):
        return self._total_data

