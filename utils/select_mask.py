import numpy as np
from sklearn.preprocessing import scale
import torch 
import scipy.io as sio
import h5py
import matplotlib.pyplot as plt

def norm(tensor, axes=(0, 1, 2), keepdims=True):

    tensor = np.linalg.norm(tensor, axis=axes, keepdims=True)

    if not keepdims: return tensor.squeeze()

    return tensor


def find_center_ind(kspace, axes=(1, 2, 3)):

    center_locs = norm(kspace, axes=axes).squeeze()

    return np.argsort(center_locs)[-1:]

def index_flatten2nd(ind, shape):

    array = np.zeros(np.prod(shape))
    array[ind] = 1
    ind_nd = np.nonzero(np.reshape(array, shape))

    return [list(ind_nd_ii) for ind_nd_ii in ind_nd]

def random_mask(self, shape):
    mask = np.zeros(shape)

    x=0
    y=0

    for x in range(0,255):
        for y in range(0,255):
            if np.random.rand() < 0.5:
                mask[x,y] = 1
            else:
                mask[x,y] = 0
    
    mask_torch = torch.from_numpy(mask).float().cuda()
    return mask_torch

class select_mask():

    def __init__(self, ratio = 0.5, small_acs_block=(4,4)):
        self.ratio = ratio
        self.small_acs_block = small_acs_block

    def guassian_selection(self, kspace, mask, std_scale=4):
        nrow, ncol = kspace.shape[0], kspace.shape[1]
        center_kx = int(find_center_ind(kspace,1))
        center_ky = int(find_center_ind(kspace,0))

        temp_mask = np.copy(mask)
        
        mask1 = np.zeros_like(mask)
        count = 0

        while count <= np.int(np.ceil(np.sum(mask[:]) * self.ratio)):
            indx = np.int(np.round(np.random.normal(loc=center_kx, scale=(nrow - 1) / std_scale)))
            indy = np.int(np.round(np.random.normal(loc=center_ky, scale=(ncol - 1) / std_scale)))\
            
            if(0<= indx < nrow and 0 <= indy < ncol and temp_mask[indx, indy] == 1):
                mask1[indx, indy] = 1
                count = count + 1
        mask1[center_kx - self.small_acs_block[0] // 2:center_kx + self.small_acs_block[0] // 2,
        center_ky - self.small_acs_block[1] // 2:center_ky + self.small_acs_block[1] // 2] = 1    
        mask2 = mask - mask1
        mask2[center_kx - self.small_acs_block[0] // 2:center_kx + self.small_acs_block[0] // 2,
        center_ky - self.small_acs_block[1] // 2:center_ky + self.small_acs_block[1] // 2] = 1

        return torch.from_numpy(mask1).float().cuda(), torch.from_numpy(mask2).float().cuda()
    
    def uniform_selection(self, kspace, mask, for_loss = False):
        nrow, ncol = kspace.shape[2], kspace.shape[3]
        center_kx = int(find_center_ind(kspace[0,0,:,:],1))
        center_ky = int(find_center_ind(kspace[0,0,:,:],0))

        temp_mask = np.copy(mask)
        
        mask1 = np.zeros_like(mask)

        pr = np.ndarray.flatten(temp_mask)
        ind = np.random.choice(np.arange(nrow * ncol),
                               size=np.int(np.count_nonzero(pr) * self.ratio), replace=False, p=pr / np.sum(pr))
        [ind_x, ind_y] = index_flatten2nd(ind, (nrow, ncol))
        
        mask1[ind_x, ind_y] = 1
        mask1[center_kx - self.small_acs_block[0] // 2:center_kx + self.small_acs_block[0] // 2,
        center_ky - self.small_acs_block[1] // 2:center_ky + self.small_acs_block[1] // 2] = 1    
        mask2_loss = mask - mask1
        mask2 = mask2_loss[center_kx - self.small_acs_block[0] // 2:center_kx + self.small_acs_block[0] // 2,
        center_ky - self.small_acs_block[1] // 2:center_ky + self.small_acs_block[1] // 2] = 1
        mask1_loss = mask - mask2
        

        return torch.from_numpy(mask1).float().cuda(), torch.from_numpy(mask2).float().cuda(), torch.from_numpy(mask1_loss).float().cuda(), torch.from_numpy(mask2_loss).float().cuda()


class create_mask():

    def __init__(self, ratio = 0.2, small_acs_block=(24,24)):
        self.ratio = ratio
        self.small_acs_block = small_acs_block

    def guassian_creation(self, shape, std_scale=4):
        nrow, ncol = shape[0], shape[1]
        center_kx = int(nrow//2)
        center_ky = int(ncol//2)

        mask1 = np.zeros(shape=shape)
        count = 0

        while count <= np.int(np.ceil(nrow*ncol) * self.ratio):
            indx = np.int(np.round(np.random.normal(loc=center_kx, scale=(nrow - 1) / std_scale)))
            indy = np.int(np.round(np.random.normal(loc=center_ky, scale=(ncol - 1) / std_scale)))
            
            if(0<= indx < nrow and 0 <= indy < ncol):
                mask1[indx, indy] = 1
                count = count + 1

        return mask1
    
    def uniform_creation(self, shape, std_scale=4, vertical=False):
        nrow, ncol = shape[0], shape[1]
        center_kx = int(nrow//2)
        center_ky = int(ncol//2)
        mask1 = np.zeros(shape=shape)
        temp_mask = np.ones(shape=(256,1))
        ratio = (self.ratio*shape[0] - self.small_acs_block[0])/shape[0]
        pr = np.ndarray.flatten(temp_mask)
        ind = np.random.choice(np.arange(nrow),
                               size=np.int(np.count_nonzero(pr) * ratio), replace=False, p=pr / np.sum(pr))
        if vertical:
            mask1[:, ind] = 1
            mask1[:,center_ky - self.small_acs_block[0] // 2:center_ky + self.small_acs_block[0] // 2] = 1
        else: 
            mask1[ind, :] = 1
            mask1[center_kx - self.small_acs_block[0] // 2:center_kx + self.small_acs_block[0] // 2,:] = 1 

        return mask1
    
    def guassian_cartesian_creation(self, shape, std_scale=4, vertical=True):
        nrow, ncol = shape[0], shape[1]
        center_ky = int(ncol//2)
        mask = np.zeros(shape, dtype=np.float32)
        ratio = (self.ratio*shape[0] - self.small_acs_block[0])/shape[0]
        nACS = self.small_acs_block[0]
        ACS_s = round((ncol - nACS)/2)
        ACS_e = ACS_s + nACS
        max_ = int(ncol * ratio)
        count = 0
        while count <= max_:
            r = np.int(np.round(np.random.normal(loc=center_ky, scale=(ncol-1) / std_scale)))
           
            if(0<= r < nrow):
                 mask[:, r] = 1
                 count = count + 1

        mask[:, ACS_s:ACS_e] = 1
        
        return mask

class fast_mri_mask():
    def __init__(self, acceleration):
        self.acceleration = acceleration
        if self.acceleration == 4:
            self.centerfraction = 0.08
        else:
            self.centerfraction = 0.04

    def equispaced_cartesian_creation(self, shape, vertical=True):
        nrow, ncol = shape[0], shape[1]

        num_low_freqs = int(round(ncol * self.centerfraction))

        mask = np.zeros(shape, dtype=np.float32)
        pad = (ncol - num_low_freqs + 1) // 2
        mask[:, pad : pad + num_low_freqs] = 1

        # determine acceleration rate by adjusting for the number of low frequencies
        adjusted_accel = (self.acceleration * (num_low_freqs - ncol)) / (
            num_low_freqs * self.acceleration - ncol
        )
        offset = np.random.randint(0, round(adjusted_accel))

        accel_samples = np.arange(offset, ncol - 1, adjusted_accel)
        accel_samples = np.around(accel_samples).astype(np.uint)
        mask[:, accel_samples] = 1


        return mask
    
    def uniform_cartesian_creation(self, shape, vertical=True):
        
        nrow, ncol = shape[0], shape[1]
        num_low_freqs = int(round(ncol * self.centerfraction))
        prob = (ncol / self.acceleration - num_low_freqs) / (
            ncol - num_low_freqs
        )
        mask = np.zeros(shape, dtype=np.float32)
        for i in range(ncol):
            if np.random.uniform() < prob:
                mask[:, i] = 1
        
        pad = (ncol - num_low_freqs + 1) // 2
        mask[:, pad : pad + num_low_freqs] = 1
        
        # mask_shape = [1 for _ in shape]
        # mask_shape[-2] = ncol

        # mask.reshape(*mask_shape).astype(np.float32)
        return mask
        
        


def index_flatten2nd(ind, shape):

    array = np.zeros(np.prod(shape))
    array[ind] = 1
    ind_nd = np.nonzero(np.reshape(array, shape))
    
    return [list(ind_nd_ii) for ind_nd_ii in ind_nd]
    

