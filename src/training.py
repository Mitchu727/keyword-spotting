import torch
from torch.nn.functional import one_hot
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn

from datasets import load_from_disk

from src.utils import get_processed_dataset_path, get_base_model_dir
from src.model import Net, NetV2


def encode_one_hot(labels):
    return one_hot(torch.tensor(labels), num_classes=36)


def evaluate_on_validation_set(model, dataloader):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in dataloader:
            mfcc, labels = batch
            mfcc = mfcc.unsqueeze(1).to(device)
            outputs = model(mfcc)
            loss = criterion(outputs, labels.to(device)).item()
            val_loss += loss
            pred = torch.max(outputs.data, 1)[1]
            correct += pred.eq(labels.to(device)).sum().item()
    accuracy = correct / len(dataloader.dataset)
    print("Validation Loss: {:.4f}".format(val_loss))
    print("Validation Accuracy: {:.2f}%".format(100 * accuracy))
    model.train()
    return val_loss, accuracy



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Assuming that we are on a CUDA machine, this should print a CUDA device:

if __name__ == "__main__":
    processed_dataset_path = str(get_processed_dataset_path())
    dataset = load_from_disk(processed_dataset_path)

    train_dataset_raw = dataset['train'].with_format("torch")
    train_dataset = data_utils.TensorDataset(train_dataset_raw["mfcc"], train_dataset_raw["label"])
    train_loader = data_utils.DataLoader(train_dataset, batch_size=256, shuffle=True)

    validation_dataset_raw = dataset['validation'].with_format("torch")
    validation_dataset = data_utils.TensorDataset(validation_dataset_raw["mfcc"], validation_dataset_raw["label"])
    validation_loader = data_utils.DataLoader(validation_dataset, batch_size=256)

    writer = SummaryWriter()

    # model = Net().to(device)
    model = NetV2().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # sample = train_dataset_raw["mfcc"][0].unsqueeze(0).unsqueeze(0).to(device)
    # print(sample.shape)
    # print(model(sample))

    for epoch in range(1000):
        running_loss = 0.0
        for batch in train_loader:
            mfcc, label = batch
            mfcc = mfcc.unsqueeze(1).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            #
            # forward + backward + optimize
            outputs = model(mfcc)
            loss = criterion(outputs, label.to(device))

            #
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # print(f"Outputs: {outputs}")
            # print(f"Labels: {label}")
            # print(f"Loss: {loss.item()}")

        print(f'Epoch [{epoch + 1}],  training_loss: {running_loss / 100:.3f}')
        val_loss, val_accuracy = evaluate_on_validation_set(model, validation_loader)
        writer.add_scalar("train/running_loss", running_loss, epoch)
        writer.add_scalar("val/validation_loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_accuracy, epoch)

    writer.flush()
    writer.close()

    torch.save(model.state_dict(), get_base_model_dir())
    model.load_state_dict(torch.load(get_base_model_dir()))  # checking if saved model can be loaded again
