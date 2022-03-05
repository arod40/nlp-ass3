import linecache

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

NUMBER_TOKENS = 3536


class CharacterLanguageModelDataset(Dataset):
    def __init__(
        self,
        data_csv,
        predict_last_only=True,
        sen_len=None,
        padded=True,
        load_label=True,
    ):
        self._data_csv = data_csv
        self._total_data = 0
        self._predict_last_only = predict_last_only
        self._padded = padded
        self._sen_len = sen_len or -1
        self._load_label = load_label
        print("Reading data...")
        with open(self._data_csv, "r") as f:
            for _ in tqdm(f.readlines()[1:]):
                self._total_data += 1
        print("DONE!")

    def __getitem__(self, idx):
        id, tokens = linecache.getline(self._data_csv, idx + 2).split(",")

        tokens = [int(t) for t in tokens.split() if self._padded or t != "0"]

        if self._load_label:
            X = torch.LongTensor(tokens[-self._sen_len - 1 : -1])
            # loading the sequence shifted to the left if predicting all the next-tokens in the sequence
            y = (
                tokens[-1]
                if self._predict_last_only
                else torch.LongTensor(tokens[-self._sen_len :])
            ) - 1
        else:
            X, y = torch.LongTensor(tokens[-self._sen_len :]), 0
        return id, (X, y)

    def __len__(self):
        return self._total_data

