import librosa
from datasets import load_dataset
from src.utils import get_processed_dataset_path


def to_mfcc(example):
    audio_array = example["audio"]["array"]
    sr = example["audio"]["sampling_rate"]
    mfcc = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=40)
    return {"mfcc": mfcc}


if __name__ == "__main__":
    dataset = load_dataset("speech_commands", "v0.02")
    processed_dataset = dataset.map(to_mfcc, num_proc=12)
    processed_dataset_path = str(get_processed_dataset_path())
    processed_dataset.save_to_disk(processed_dataset_path)