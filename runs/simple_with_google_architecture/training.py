from pathlib import Path

import torch
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn

from datasets import load_from_disk

from src.utils import get_processed_dataset_path, get_run_dir, save_training_and_models_file_to_run_folder, evaluate_on_dataset
from src.models import Net, get_models_path
import uuid


def setup_experiment():
    experiment_id = str(uuid.uuid4())
    run_dir = get_run_dir(experiment_id)
    writer = SummaryWriter(str(run_dir))
    save_training_and_models_file_to_run_folder(run_dir, get_training_path(), get_models_path())
    return writer


def end_experiment(writer, model):
    writer.flush()
    writer.close()
    model_path = Path(writer.get_logdir()) / "model"
    torch.save(model.state_dict(), model_path)
    model.load_state_dict(torch.load(model_path))


def prepare_dataloaders():
    processed_dataset_path = str(get_processed_dataset_path())
    dataset = load_from_disk(processed_dataset_path)

    train_dataset_raw = dataset['train'].with_format("torch")
    train_dataset = data_utils.TensorDataset(train_dataset_raw["mfcc"], train_dataset_raw["label"])
    train_loader = data_utils.DataLoader(train_dataset, batch_size=256, shuffle=True)

    validation_dataset_raw = dataset['validation'].with_format("torch")
    validation_dataset = data_utils.TensorDataset(validation_dataset_raw["mfcc"], validation_dataset_raw["label"])
    validation_loader = data_utils.DataLoader(validation_dataset, batch_size=256)

    test_dataset_raw = dataset['test'].with_format("torch")
    test_dataset = data_utils.TensorDataset(test_dataset_raw["mfcc"], test_dataset_raw["label"])
    test_loader = data_utils.DataLoader(test_dataset, batch_size=256)
    return train_loader, validation_loader, test_loader


def get_training_path() -> Path:
    return Path(__file__)


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    writer = setup_experiment()

    train_loader, validation_loader, test_loader = prepare_dataloaders()

    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1000):
        running_loss = 0.0
        for batch in train_loader:
            mfcc, label = batch
            mfcc = mfcc.unsqueeze(1).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(mfcc)
            loss = criterion(outputs, label.to(device))
            loss.backward()
            optimizer.step()

            # update statistics
            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}],  training_loss: {running_loss / 100:.3f}')
        writer.add_scalar("train/running_loss", running_loss, epoch)

        val_loss, val_accuracy = evaluate_on_dataset(model, validation_loader, criterion, device)
        print("Validation Loss: {:.4f}".format(val_loss))
        print("Validation Accuracy: {:.2f}%".format(100 * val_accuracy))

        writer.add_scalar("val/validation_loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_accuracy, epoch)

    test_loss, test_accuracy = evaluate_on_dataset(model, test_loader, criterion, device)
    print("Test Loss: {:.4f}".format(test_loss))
    print("Test Accuracy: {:.2f}%".format(100 * test_accuracy))

    writer.add_scalar("test/test_loss", test_loss)
    writer.add_scalar("test/accuracy", test_accuracy)

    end_experiment(writer, model)
