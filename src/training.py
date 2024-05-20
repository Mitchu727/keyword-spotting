import numpy
from datasets import load_from_disk

from src.utils import get_processed_dataset_path
from src.dataset_preprocessing import pad_with_zeroes

from torch.utils.data import DataLoader

if __name__ == "__main__":
    processed_dataset_path = str(get_processed_dataset_path())
    dataset = load_from_disk(processed_dataset_path)
    sample = dataset['train'][0]
    mfcc = numpy.array(sample['mfcc'])
    print(mfcc.shape)

    train_dataset = dataset['train']
    dataloader = DataLoader(train_dataset['mfcc'], batch_size=4)
    for epoch in range(2):
        for batch in dataloader:
            print(batch)

