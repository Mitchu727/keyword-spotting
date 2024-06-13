import librosa
import torch
from datasets import load_dataset
from src.utils import get_processed_dataset_path
import numpy as np
from torchaudio import transforms


def pad_with_zeroes(desired_size_length, array):
    padded_array = np.zeros((array.shape[0], desired_size_length))
    padded_array[:, :array.shape[1]] = array[:, :desired_size_length]
    return padded_array


# transform = transforms.MFCC(
#     sample_rate=16000,
#     n_mfcc=40
# )

def to_mfcc(example):
    audio_array = example["audio"]["array"]
    sr = example["audio"]["sampling_rate"]
    mfcc = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=40)
    mfcc /= 1000
    mfcc = pad_with_zeroes(32, mfcc)
    return {"mfcc": mfcc}


if __name__ == "__main__":
    dataset = load_dataset("speech_commands", "v0.02")

    # sample = dataset["train"][0]
    # print(to_mfcc(sample))
    #
    processed_dataset = dataset.map(to_mfcc, num_proc=12)
    processed_dataset_path = str(get_processed_dataset_path())
    processed_dataset.save_to_disk(processed_dataset_path)