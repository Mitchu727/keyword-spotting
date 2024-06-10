import numpy
import torch
from datasets import load_from_disk, Dataset, Features, Array2D

from src.utils import get_processed_dataset_path
from src.dataset_preprocessing import pad_with_zeroes
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from src.model import Net

from torch.nn.functional import one_hot
import torch.optim as optim
import torch.nn as nn


def encode_one_hot(labels):
    return one_hot(torch.tensor(labels), num_classes=36)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Assuming that we are on a CUDA machine, this should print a CUDA device:

if __name__ == "__main__":
    processed_dataset_path = str(get_processed_dataset_path())
    dataset = load_from_disk(processed_dataset_path)
    train_dataset = dataset['train'].with_format("torch")
    # features = Features({"data": Array2D(shape=(40, 60), dtype='float64')})
    # ds = Dataset.from_dict({"data": train_dataset["mfcc"]}, features=features)
    # ds = ds.with_format("torch")

    writer = SummaryWriter()

    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # train_dataset = Dataset.from_dict({"data": dataset['train']["mfcc"], "labels": dataset["train"]["label"]})

    # mfcc = dataset['train']["mfcc"]
    # dataloader = DataLoader(ds, batch_size=16, num_workers=4)
    print(train_dataset["mfcc"])
    print(train_dataset["label"])

    import torch.utils.data as data_utils
    # dataloader = DataLoader(train_dataset["label"], batch_size=4)
    train = data_utils.TensorDataset(train_dataset["mfcc"], train_dataset["label"])
    train_loader = data_utils.DataLoader(train, batch_size=64)
    
    for epoch in range(100):
        running_loss = 0.0
        i = 0
        for batch in train_loader:
            mfcc, label = batch
            # print(mfcc.shape)
            # data = torch.tensor(batch)
            # # data, labels = batch
            mfcc = mfcc.unsqueeze(1).to(device)
            # labels = encode_one_hot(label).to(device)
            # print(batch)
            # # zero the parameter gradients
            optimizer.zero_grad()
            #
            # # forward + backward + optimize
            outputs = net(mfcc)
            loss = criterion(outputs, label.to(device))
            writer.add_scalar("Loss/train", loss, epoch)
            #
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 100 == 99:    # print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
            i += 1

    writer.flush()
    writer.close()
