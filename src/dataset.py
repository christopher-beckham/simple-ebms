import numpy as np
from scipy import stats
from sklearn.datasets import make_swiss_roll
import torch
from torch.utils.data import Dataset, TensorDataset
from typing import Tuple

from .util import rejection_sample_dataset

# Sample a batch from the swiss roll
def sample_swiss_roll(size, noise=1.0):
    x, _= make_swiss_roll(size, noise=noise)
    return x[:, [0, 2]] / 10.0

def get_dataset() -> Tuple[Dataset, Dataset]:
    data_x = sample_swiss_roll(size=10**4)
    origin = np.asarray([[0.0, 0.0]])
    data_y = np.sum( np.exp( -(data_x - origin)**2 ), axis=1, keepdims=True)
    ind = rejection_sample_dataset(
        data_x,
        density_thresh=0.08, 
        accept_p=0.1
    )
    train_dataset = TensorDataset(
        torch.from_numpy(data_x[ind]).float(), 
        torch.from_numpy(data_y[ind]).float()
    )
    valid_dataset = TensorDataset(
        torch.from_numpy(data_x[~ind]).float(), 
        torch.from_numpy(data_y[~ind]).float()
    )
    return train_dataset, valid_dataset

if __name__ == '__main__':
    train_dataset, _ = get_dataset(
        density_thresh=0.08, accept_p=0.1
    )
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32)
    print(iter(train_loader).__next__())
