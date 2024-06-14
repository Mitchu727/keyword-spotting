from pathlib import Path

import librosa
import torch
from matplotlib import pyplot as plt


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_processed_dataset_path() -> Path:
    return get_project_root() / "datasets" / "speech_commands-processed-v1"


def get_base_model_dir() -> Path:
    return get_project_root() / "models"


def get_run_dir(experiment_id) -> Path:
    return get_project_root() / "runs" / experiment_id


def visualise(path: str, mfccs) -> None:
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
    fig.colorbar(img, ax=[ax])
    ax.set(title='MFCC')
    plt.savefig(path)


def save_training_and_models_file_to_run_folder(run_dir: Path, training_path: Path, model_path: Path, ) -> None:
    model_save_path = run_dir / "models.py"
    training_save_path = run_dir / "training.py"
    with open(training_path, "r") as f:
        training = f.read()
    with open(training_save_path, "w") as f:
        f.write(training)
    with open(model_path, "r") as f:
        model = f.read()
    with open(model_save_path, "w") as f:
        f.write(model)


def evaluate_on_dataset(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in dataloader:
            mfcc, labels = batch
            mfcc = mfcc.unsqueeze(1).to(device)
            outputs = model(mfcc)
            loss = criterion(outputs, labels.to(device)).item()
            running_loss += loss
            pred = torch.max(outputs.data, 1)[1]
            correct += pred.eq(labels.to(device)).sum().item()
    accuracy = correct / len(dataloader.dataset)
    model.train()
    return running_loss, accuracy
