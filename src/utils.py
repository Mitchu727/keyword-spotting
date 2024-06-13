from pathlib import Path

import librosa
from matplotlib import pyplot as plt


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_processed_dataset_path() -> Path:
    return get_project_root() / "datasets" / "speech_commands-processed-v1"


def get_base_model_dir() -> Path:
    return get_project_root() / "models" / "base_speech_commands"

def visualise(path: str, mfccs) -> None:
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
    fig.colorbar(img, ax=[ax])
    ax.set(title='MFCC')
    plt.savefig(path)