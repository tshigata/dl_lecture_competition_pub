import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from scipy.signal import resample, butter, filtfilt
# import PreProcess


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", preprocess: bool = True) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.preprocess = preprocess
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."
        # if self.preprocess:
        #     self.X = self.preprocess_data(self.X)

    def preprocess_data(self, data):
        # Example preprocessing steps
        data = self.resample_data(data, new_rate=128)
        data = self.bandpass_filter(data, lowcut=0.5, highcut=50, fs=128)
        data = self.standardize(data)
        data = self.baseline_correction(data, baseline_period=50)
        return data
    
    def resample_data(self, data, new_rate):
        return resample(data, new_rate, axis=-1)

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data, axis=-1)
        return y

    def standardize(self, data):
        mean = np.mean(data, axis=-1, keepdims=True)
        std = np.std(data, axis=-1, keepdims=True)
        return (data - mean) / std

    def baseline_correction(self, data, baseline_period):
        baseline = np.mean(data[:, :, :baseline_period], axis=-1, keepdims=True)
        return data - baseline
    
    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]