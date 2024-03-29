import torch.nn as nn


class CharacterLanguageModel(nn.Module):
    def __init__(
        self,
        no_characters,
        embedding_dim,
        no_layers,
        lstms_hidden,
        lstm_out,
        last_time_step_only=True,
    ):
        super().__init__()

        self.embedding = nn.Embedding(no_characters + 1, embedding_dim)
        self._last_only = last_time_step_only

        class DropHidden(nn.Module):
            def __init__(self, lstm):
                super().__init__()
                self.lstm = lstm

            def forward(self, X):
                X, _ = self.lstm(X)
                return X

        sizes = [embedding_dim, *[lstms_hidden] * no_layers, lstm_out]
        self.lstms = nn.Sequential(
            *[
                DropHidden(nn.LSTM(_in, _out, batch_first=True))
                for _in, _out in zip(sizes[:-1], sizes[1:])
            ]
        )
        self.output_layer = nn.Linear(lstm_out, no_characters)

    @property
    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, X):
        X = self.embedding(X)
        X = self.lstms(X)
        if self._last_only:
            X = X[:, -1]  # taking last time step
        X = self.output_layer(X)

        return X
