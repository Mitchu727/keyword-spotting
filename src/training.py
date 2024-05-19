import numpy
from datasets import load_from_disk

from src.utils import get_processed_dataset_path

if __name__ == "__main__":
    processed_dataset_path = str(get_processed_dataset_path())
    dataset = load_from_disk(processed_dataset_path)
    sample = dataset['train'][0]
    mfcc = numpy.array(sample['mfcc'])
    print(mfcc.shape)

