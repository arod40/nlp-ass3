from data_utils import CharacterLanguageModelDataset, NUMBER_TOKENS
from models import CharacterLanguageModel
from torch.utils.data import DataLoader


dataset = CharacterLanguageModelDataset("data/seq_train.csv")
data_loader = DataLoader(dataset, batch_size=1)
model = CharacterLanguageModel(NUMBER_TOKENS, 300, 3, 100, 50)
print(model.number_of_parameters)

for id, (X, y) in data_loader:
    print(X.shape)
    print(y.shape)
    print(model(X))
    break

