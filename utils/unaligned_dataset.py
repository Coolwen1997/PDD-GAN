from os import name
import os.path
from torch._C import device

from torch.utils.data.dataset import Dataset
from PIL import Image
import random
import logging
import torch
import h5py
import ismrmrd.ismrmrd.xsd as xsd
import numpy as np
import pickle
import pathlib
from os import path, listdir
from os.path import splitext
from utils.math import *



class fastmri_dataset(Dataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opts, mask_up, mask_down, mask_under, mode):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.mode = mode
        if self.mode == 'TRAIN':
            self.dir_A = os.path.join(opts.data_root, 'TRAIN_A')  # create a path '/path/to/data/trainA'
            self.dir_B = os.path.join(opts.data_root, 'TRAIN_B')  # create a path '/path/to/data/trainB'
            self.slice_range = [0,12]
            self.data_A_list = self.get_data(self.dir_A)
            self.data_B_list = self.get_data(self.dir_B)

        if self.mode == 'VAL':
            self.dir_A = os.path.join(opts.data_root, "VAL_B")
            self.dir_B = os.path.join(opts.data_root, "VAL_A")
            self.slice_range = [0,12]
            self.data_A_list = self.get_data(self.dir_A)
            self.data_B_list = self.get_data(self.dir_B)
        
        self.resolution = 256
        self.model = 'singlecoil' 
        self.recons_key = 'reconstruction_esc'
        
        
        
        self.data_size = min(len(self.data_A_list), len(self.data_B_list))

        if self.mode == 'TRAIN':
            logging.info(f'Creating training dataset with {self.data_size} examples')
        else:
            logging.info(f'Creating validation dataset with {self.data_size} examples')
        self.mask_up = mask_up
        self.mask_down = mask_down
        self.mask_under = mask_under
        self.device = torch.device("cuda:0")
        


    def __len__(self):
        return self.data_size

    def norm(self,x):
        max = torch.max(x)
        min = torch.min(x)
        x = (x-min)/(max-min)
        return x
    
    def crop_toshape(self,x):
        cropx = (x.shape[-2] - self.resolution) // 2
        cropy = (x.shape[-1] - self.resolution) // 2
        x = x[:, cropx:(cropx+self.resolution),cropy:(cropy+self.resolution)]
        return x
    

    # @classmethod
    def preprocess(self,kspace):
        # split to real and imaginary channels
        target_img = ifft(kspace)
        
        label = self.norm(target_img)
        # label = target_img
        kspace = fft(label)
        
        masked_up_kspace = (self.mask_up * kspace.permute(1,2,0)).permute(2,0,1)
        
        zf_img = ifft(masked_up_kspace)
        # label = ifft(kspace)
        return kspace, masked_up_kspace, zf_img, label

        
        
        
    def get_data(self,path):
        self.examples = []
        files = list(pathlib.Path(path).iterdir())

        for fname in sorted(files):
            self.examples += [(fname, slice) for slice in range(self.slice_range[0], self.slice_range[1])]
        
        return self.examples
    
    def get_data_dict(self, example):
        fname, slice = example
        data_dict = {}
        with h5py.File(fname, 'r') as data:
            target_img = data[self.recons_key][slice]
            kspace = torch.from_numpy(data['kspace'][slice])
        
        # kspace = self.crop_toshape(kspace)
        kspace = torch.stack((kspace.real, kspace.imag), dim=0)
        kspace, masked_kspace, zf_img, target_img = self.preprocess(kspace)

        data_dict['kspace'] = kspace
        data_dict['masked_kspace'] = masked_kspace
        data_dict['zf_img'] = zf_img
        data_dict['target_img'] = target_img
        return data_dict

    def __getitem__(self, i):
        
        A_dict = self.get_data_dict(self.data_A_list[i])
        
        B_dict = self.get_data_dict(self.data_B_list[i])

        under_k = A_dict['masked_kspace']
        
        under_img = A_dict['zf_img']

        full_img_A = A_dict['target_img']
        # under_i_reference = under_img

        full_k_A = A_dict['kspace']
        full_k = B_dict['kspace']
        full_img = B_dict['target_img']
        

        return {'under_k_A': under_k, 'full_k_B': full_k,'under_img_A': under_img, 'full_img_B':full_img,\
                    'target_img_A':full_img_A, 'target_k_A':full_k_A}
    
class IXI_dataset(Dataset):
    def __init__(self, opts, mask_up, mask_down, mask_under, mode):
        
        self.mode = mode
        
        self.num_input_slices = 3
        self.resolution = 256
        self.minmax_noise_val = [-0.01, 0.01]

        if self.mode == 'TRAIN':
            self.dir_A = os.path.join(opts.data_root, 'TRAIN_A')  # create a path '/path/to/data/trainA'
            self.dir_B = os.path.join(opts.data_root, 'TRAIN_B')  # create a path '/path/to/data/trainB'
            slice_range = [40,100]
            self.slice_range = slice_range
            self.data_A_list = self.get_data(self.dir_A)
            self.data_B_list = self.get_data(self.dir_B)

        if self.mode == 'VAL':
            self.dir_A = os.path.join(opts.data_root, "VAL_B")
            self.dir_B = os.path.join(opts.data_root, "VAL_A")
            slice_range = [40,100]
            self.slice_range = slice_range
            self.data_A_list = self.get_data(self.dir_A)
            self.data_B_list = self.get_data(self.dir_B)

        if self.mode == 'TEST':
            self.dir_A = os.path.join(opts.data_root, "TEST_A")
            self.dir_B = os.path.join(opts.data_root, "TEST_B")
        
        

        self.data_size = len(max(self.data_A_list, self.data_B_list))

        if self.mode == 'TRAIN':
            logging.info(f'Creating training dataset with {self.data_size} examples')
        else:
            logging.info(f'Creating validation dataset with {self.data_size} examples')
        self.mask_up = mask_up
        self.mask_down = mask_down
        self.mask_under = mask_under
        

    def __len__(self):
        return self.data_size

    def crop_toshape(self,x):
        assert 0 < self.resolution <= x.shape[-2]
        assert 0 < self.resolution <= x.shape[-1]
        cropx = (x.shape[-2] - self.resolution) // 2
        cropy = (x.shape[-1] - self.resolution) // 2
        x = x[cropx:(cropx+self.resolution),cropy:(cropy+self.resolution)]
        return x
    
    def get_data(self, data_dir):
        if self.mode == 'TRAIN':
            num = 50
            file_names = [splitext(file)[0] for file in listdir(data_dir)[:num+1]
                    if not file.startswith('.')]
        elif self.mode == 'VAL':
            num = 10
            file_names = [splitext(file)[0] for file in listdir(data_dir)
                    if not file.startswith('.')]
            
        
        
        data_list = []

        for file_name in file_names:
            try:
                full_file_path = path.join(data_dir,file_name+'.hdf5')
                with h5py.File(full_file_path, 'r') as f:
                    numOfSlice = f['target_img'].shape[0]
                if numOfSlice < self.slice_range[1]:
                    continue
                for slice in range(self.slice_range[0], self.slice_range[1]):
                    data_list.append((file_name, slice))
            except:
                continue
        
        return data_list

    def get_data_dict(self,data_dir, file_name, slice_num):
        file_path = path.join(data_dir,file_name + '.hdf5')

        with h5py.File(file_path, 'r') as f:
            # zf_img = f['zf_img'][slice_num]
            target_img = f['target_img'][slice_num]
            target_kspace = f['kspace'][slice_num]
            # masked_kspace = f['masked_kspace'][slice_num]
        
        #DIIK
        target_img = torch.from_numpy(target_img)
        imag = torch.zeros_like(target_img) 
        target_img = torch.cat((target_img, imag), 0)
        target_kspace = fft(target_img)
        masked_up_kspace = (self.mask_up * target_kspace.permute(1,2,0)).permute(2,0,1)
        zf_img = ifft(masked_up_kspace)

        data_dict = {}
        data_dict['kspace'] = target_kspace
        data_dict['masked_kspace'] = masked_up_kspace
        data_dict['zf_img'] = zf_img
        data_dict['target_img'] = target_img
        
        return data_dict, file_path


    def __getitem__(self, i):
        A_file_name, A_slice_num = self.data_A_list[i]

        A_dict, A_path = self.get_data_dict(self.dir_A, A_file_name, A_slice_num)

        B_file_name, B_slice_num = self.data_B_list[i]

        B_dict, B_path = self.get_data_dict(self.dir_B, B_file_name, B_slice_num)
        
        under_k = A_dict['masked_kspace']
        
        under_img = A_dict['zf_img']
        full_img_A = A_dict['target_img']
        full_k_A = A_dict['kspace']

        full_k = B_dict['kspace']
        full_img = B_dict['target_img']

        return {'under_k_A': under_k, 'full_k_B': full_k,'under_img_A': under_img, 'full_img_B':full_img,\
                    'target_img_A':full_img_A, 'target_k_A':full_k_A}
    


