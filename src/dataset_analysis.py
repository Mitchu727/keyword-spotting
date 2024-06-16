import librosa
import matplotlib.pyplot as plt
from datasets import load_dataset

from src.dataset_preprocessing import pad_with_zeroes
from src.utils import get_processed_dataset_path
import numpy as np

def print_dictionary(dictionary):
    for key, value in dictionary.items():
        print(f'{key}, {value}')


def increment_dict_with_key(dictionary, key):
    if key not in dictionary.keys():
        dictionary[key] = 1
    else:
        dictionary[key] += 1
    return dictionary


def mfcc_length(example):
    audio_array = example["audio"]["array"]
    sr = example["audio"]["sampling_rate"]
    mfcc = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=40)
    return mfcc.shape[1]


def count_sr_in_dataset(dataset):
    sr_counts = {}
    for sample in dataset:
        key = str(sample["audio"]["sampling_rate"])
        sr_counts = increment_dict_with_key(sr_counts, key)
    return sr_counts


def count_labels_in_dataset(dataset):
    labels_counts = {}
    for sample in dataset:
        key = str(sample["label"])
        labels_counts = increment_dict_with_key(labels_counts, key)
    return labels_counts


def count_mfcc_lengths_in_dataset(dataset):
    mfcc_length_counts = {}
    for sample in dataset:
        sample_mfcc_length = mfcc_length(sample)
        key = str(sample_mfcc_length)
        mfcc_length_counts = increment_dict_with_key(mfcc_length_counts, key)
    return mfcc_length_counts


if __name__ == "__main__":
    dataset = load_dataset("speech_commands", "v0.02")

    analysed_datasets_names = ["train", "validation", "test"]
    for analysed_dataset_name in analysed_datasets_names:
        print(f"Zbiór: {analysed_dataset_name}")
        analysed_dataset = dataset[analysed_dataset_name]
        print(f"Ilość spektogramów o danej wielkości w formacie wielkość: ilość próbek")
        print_dictionary(count_mfcc_lengths_in_dataset(analysed_dataset))
        print(f"Częstotliwość samplowania w formacie częstotliwość: ilość próbek")
        print_dictionary(count_sr_in_dataset(analysed_dataset))
        print(f"Ilość klas w formacie klasa: ilość próbek")
        print_dictionary(count_labels_in_dataset(analysed_dataset))
    # print(count_sr_in_dataset(dataset["train"]))
    # print(count_sr_in_dataset(dataset["validation"]))
    # print(count_sr_in_dataset(dataset["test"]))
    # processed_dataset = dataset.map(to_mfcc, num_proc=12)