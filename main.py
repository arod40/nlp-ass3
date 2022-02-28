from datetime import datetime
from math import floor, inf
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data_utils import NUMBER_TOKENS, CharacterLanguageModelDataset
from models import CharacterLanguageModel


def train(model, data, criterion, optimizer):
    model.train()

    loss_acc = 0
    for _, (X, y) in tqdm(data):
        print("X", X.shape)
        print("y", y.shape)
        optimizer.zero_grad()

        out = model(X)
        print("out", out.shape)

        input()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        loss_acc += loss.item()
    return loss_acc / len(data)


def evaluate(model, data, metrics):
    model.eval()

    results = {name: 0 for name, _ in metrics}
    for _, (X, y) in tqdm(data):
        out = model(X)
        for metric in metrics:
            name, criterion = metric
            loss = criterion(out, y)
            results[name] += loss.item()

    for name, _ in metrics:
        results[name] /= len(data)

    return results


# To work correctly as accuracy, validation data must set batch_size=1
def correct_count(out, target):
    return (out.argmax(dim=1) == target).sum()


train_ratio = 0.7
val_ratio = 0.2
epochs = 10
checkout = 2
save_path = Path(f"models/exp-{str(datetime.now().timestamp())}/")
save_model_checkout = True
save_path.mkdir()


dataset = CharacterLanguageModelDataset("data/seq_train.csv")
train_size = floor(train_ratio * len(dataset))
val_size = floor(val_ratio * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)


train_data = DataLoader(train_dataset, batch_size=100)
train_data_batch1 = DataLoader(train_dataset, batch_size=1)
eval_data = DataLoader(val_dataset, batch_size=1)
test_data = DataLoader(test_dataset, batch_size=1)

print("Training examples:", train_size)
print("Validation examples:", val_size)
print("Testing examples:", test_size)


model = CharacterLanguageModel(NUMBER_TOKENS + 1, 300, 3, 100, 50)
criterion = nn.NLLLoss()
optimizer = Adam(model.parameters())
print("Number of parameters:", model.number_of_parameters)


history = []
best_value = inf
eval_metrics = [("loss", criterion, "accuracy", correct_count)]
cross_validation_metric = "accuracy"


print("##########################")
print("TRAINING...")
print("##########################")

for epoch in range(epochs):
    print(f"Epoch {epoch}")
    loss = train(model, train_data, criterion, optimizer)
    history.append(loss)

    if epoch % checkout == 0:
        print("Evaluating the model...")
        print("...on train data")
        train_results = evaluate(model, train_data_batch1, eval_metrics)
        print("...on validation data")
        eval_results = evaluate(model, eval_data, eval_metrics)
        for name, value in train_results.items():
            print(f"Train {name}: {value}")
        for name, value in eval_results.items():
            print(f"Eval {name}: {value}")

        if save_model_checkout:
            print("Saving current state...")
            torch.save(model.state_dict(), save_path / f"epoch-{epoch}.pt")

        if eval_results[cross_validation_metric] < best_value:
            print("Saving best model...")
            best_value = eval_results[cross_validation_metric]
            torch.save(model.state_dict(), save_path / "best.pt")

torch.save(history, save_path / ".history")


print("##########################")
print("TRAINING FINISHED")
print("##########################")


print("Computing test scores...")
model.load_state_dict(torch.load(save_path / "best.pt"))

metrics = evaluate(model, test_data, eval_metrics)
for name, value in metrics.items():
    print(f"Test {name}: {value}")

