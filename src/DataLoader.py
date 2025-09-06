# coding=utf-8
import os

import numpy as np
import torch as th
import json
import torch
import hdf5storage
import random
import math
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, X_train):

        self.X_train = X_train

    def __len__(self):

        return self.X_train.shape[0]

    def __getitem__(self, idx):

        return self.X_train[idx].unsqueeze(0)


class CSIDataset_Single(Dataset):
    def __init__(self, X_train):
        self.X_train = X_train

    def __len__(self):

        return self.X_train.shape[0]

    def __getitem__(self, idx):

        return self.X_train[idx]
    


def data_load_single(args, dataset): # 加载单个数据集

    folder_path_test = '../dataset/{}/X_test.mat'.format(dataset)

    X_test = hdf5storage.loadmat(folder_path_test)
    X_test_complex = torch.tensor(np.array(X_test['X_val'], dtype=complex)).unsqueeze(1)

    ## 实部+虚部
    X_test = torch.cat((X_test_complex.real, X_test_complex.imag),dim=1).float()

    test_data = MyDataset(X_test)


    batch_size = args.batch_size
    test_data = th.utils.data.DataLoader(test_data, num_workers=16, batch_size=batch_size, shuffle=False, pin_memory=True, prefetch_factor=4)

    return test_data


def data_load_single_zeroshot(args, dataset_name, SNR=20, dataset_type='test'): # 加载单个数据集

    folder_path = os.path.join("/home/zhangchenyu/data/csidata/zeroshot_new", f'{dataset_name}/{dataset_type}_data.mat')

    H_data = hdf5storage.loadmat(folder_path)[f'H_{dataset_type}']
    H_data = H_data.transpose(0, 1, 3, 2) #(B, T, K, U) -> (B, T, U, K)

    power = np.mean(np.abs(H_data)**2, axis=(1, 2, 3), keepdims=True)
    H_data = H_data / (np.sqrt(power) + 1e-8) 

    H_data_complex = torch.tensor(np.array(H_data, dtype=complex)).unsqueeze(1)
    H_data = torch.cat((H_data_complex.real, H_data_complex.imag),dim=1).float()

    dataset_test = CSIDataset_Single(H_data)

    data_loader = torch.utils.data.DataLoader(
        dataset_test,
        shuffle=False, 
        batch_size=args.batch_size,
        num_workers=16,
        pin_memory=True,
        # prefetch_factor=8,
    )
    return data_loader


def data_load(args, test_type='normal'):

    test_data_all = []
        
    for dataset_name in args.dataset.split('*'):
        print(f"Processing {dataset_name} for test")
        if test_type == 'normal':
            # test_data, _ = data_load_single(args, dataset_name, dataset_type=dataset_type)
            test_data = data_load_single_zeroshot(args, dataset_name)
        elif test_type == 'wifo':
            test_data = data_load_single(args, dataset_name)
        test_data_all.append(test_data)
    
    return test_data_all


def data_load_main(args, test_type='normal'):

    test_data = data_load(args, test_type)

    return test_data


def generate_gaussian_noise(data, snr_db):
    axes = tuple(range(1, data.ndim))
    signal_power = np.mean(np.abs(data) ** 2, axis=axes, keepdims=True)
    
    # Convert SNR to linear scale
    snr_linear = 10 ** (snr_db / 10)
    
    # Ensure SNR has proper shape for broadcasting
    if not isinstance(snr_linear, np.ndarray):
        snr_linear = np.array(snr_linear)
    if snr_linear.ndim == 0 or snr_linear.size == 1:
        snr_linear = snr_linear.reshape((-1,) + (1,)*(data.ndim-1))
    else:
        snr_linear = snr_linear.reshape((-1,) + (1,)*(data.ndim-1))
    
    # Calculate noise power
    noise_power = signal_power / snr_linear
    
    # Generate complex Gaussian noise
    # Real and imaginary parts scaled appropriately
    noise_real = np.random.standard_normal(data.shape) * np.sqrt(noise_power / 2)
    noise_imag = np.random.standard_normal(data.shape) * np.sqrt(noise_power / 2)
    
    # Combine into complex noise
    noise = noise_real + 1j * noise_imag
    
    return noise
