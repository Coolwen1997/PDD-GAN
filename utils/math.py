from matplotlib.pyplot import flag
import torch
import torch.nn as nn
import numpy as np

from typing import List, Optional, Tuple

# 原傅里叶变换 适用于IXI数据
def fft(input, dc=False):
    # (N, 2, W, H) -> (N, W, H, 2)
    if len(input.shape) == 4:
        input = input.permute(0, 2, 3, 1)
    if input.shape[-1] == 1:
        imag = torch.zeros_like(input) 
        input = torch.cat((input, imag),-1)

    input = torch.fft(input, 2)
    
    # (N, W, H, 2) -> (N, 2, W, H)
    if len(input.shape) == 4:
        input = input.permute(0, 3, 1, 2)
        input = fftshift2(input)
    else:
        input = input.permute(2,0,1)
        input = fftshift(input)
    return input

def ifft(input, dc=False):
    if len(input.shape) == 4:
        input = ifftshift2(input)
        input = input.permute(0, 2, 3, 1)
        
    else:
        input = ifftshift(input)
        input = input.permute(1,2,0)

    input = torch.ifft(input, 2)

    if len(input.shape) == 4:
        input = input.permute(0, 3, 1, 2)
        input = input[:,0:1,:,:]
    if len(input.shape) ==3:
        input = input.permute(2, 0, 1)
        input = input[0:1,:,:]
    return input

def fftshift(img):
    
    S = int(img.shape[-1]/2)
    img2 = torch.zeros_like(img)
    img2[:, :S, :S] = img[:, S:, S:]
    img2[:, S:, S:] = img[:, :S, :S]
    img2[:, :S, S:] = img[:, S:, :S]
    img2[:, S:, :S] = img[:, :S, S:]
    
    return img2

def fftshift2(img):
    
    S = int(img.shape[-1]/2)
    img2 = torch.zeros_like(img)
    img2[:, :, :S, :S] = img[:, :, S:, S:]
    img2[:, :, S:, S:] = img[:, :, :S, :S]
    img2[:, :, :S, S:] = img[:, :, S:, :S]
    img2[:, :, S:, :S] = img[:, :, :S, S:]
    return img2

def ifftshift2(img):
    
    S = int(img.shape[-1]/2)
    img2 = torch.zeros_like(img)
    img2[:, :, S:, S:] = img[:, :, :S, :S]
    img2[:, :, :S, :S] = img[:, :, S:, S:]
    img2[:, :, S:, :S] = img[:, :, :S, S:]
    img2[:, :, :S, S:] = img[:, :, S:, :S]
    return img2

def ifftshift(img):
    
    S = int(img.shape[-1]/2)
    img2 = torch.zeros_like(img)
    img2[ :, S:, S:] = img[ :, :S, :S]
    img2[ :, :S, :S] = img[ :, S:, S:]
    img2[ :, S:, :S] = img[ :, :S, S:]
    img2[ :, :S, S:] = img[ :, S:, :S]
    return img2

# 傅里叶变换 适用于fastmri
def fft(input):
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.

    Returns:
        The FFT of the input.
    """
    if len(input.shape) == 4:
        input = input.permute(0, 2, 3, 1)
    
    if len(input.shape) == 3:
        input = input.permute(1, 2, 0)

    if not input.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    input = ifftshift(input, dim=(-3, -2))
    input = torch.fft(input, 2, normalized=False) # normalized = True or False
    input = fftshift(input, dim=(-3, -2))

    if len(input.shape) == 4:
        input = input.permute(0, 3, 1, 2)
    
    if len(input.shape) == 3:
        input = input.permute(2, 0 ,1)

    return input

def ifft(input):

    if len(input.shape) == 4:
        input = input.permute(0, 2, 3, 1)
    
    if len(input.shape) == 3:
        input = input.permute(1, 2, 0)
        
    if not input.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    input = ifftshift(input, dim=(-3, -2))
    input = torch.ifft(input, 2, normalized=False) # normalized = True or False
    input = fftshift(input, dim=(-3, -2))

    if len(input.shape) == 4:
        input = input.permute(0, 3, 1, 2)
    
    if len(input.shape) == 3:
        input = input.permute(2, 0 ,1)

    return input


def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """
    Similar to roll but for only one dim.

    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.

    Returns:
        Rolled version of x.
    """
    shift = shift % x.size(dim)
    if shift == 0:
        return x

    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def roll(
    x: torch.Tensor,
    shift: List[int],
    dim: List[int],
) -> torch.Tensor:
    """
    Similar to np.roll but applies to PyTorch Tensors.

    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.

    Returns:
        Rolled version of x.
    """
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    for (s, d) in zip(shift, dim):
        x = roll_one_dim(x, s, d)

    return x


def fftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors

    Args:
        x: A PyTorch tensor.
        dim: Which dimension to fftshift.

    Returns:
        fftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = x.shape[dim_num] // 2

    return roll(x, shift, dim)


def ifftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors

    Args:
        x: A PyTorch tensor.
        dim: Which dimension to ifftshift.

    Returns:
        ifftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = (x.shape[dim_num] + 1) // 2

    return roll(x, shift, dim)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

def complex_abs_eval(data):
    assert data.size(1) == 2
    data = (data[:, 0:1, :, :] ** 2 + data[:, 1:2, :, :] ** 2).sqrt()
    return data[0]


def crop_toshape(x, resolution):
    cropx = (x.shape[-2] - resolution) // 2
    cropy = (x.shape[-1] - resolution) // 2
    x = x[:,:,cropx:(cropx+resolution),cropy:(cropy+resolution)]
    return x


def vis_img(img):
    img = img[0].detach().cpu().float().numpy()
    image_numpy = abs(img)
    
    return image_numpy

def tensor_to_complex_np(data: torch.Tensor):
    """
    Converts a complex torch tensor to numpy array.

    Args:
        data: Input data to be converted to numpy.

    Returns:
        Complex numpy version of data.
    """
    data.permute(0,2,3,1)
    data = data.numpy()

    return data[..., 0] + 1j * data[..., 1]

def abs(x):
    y = np.abs(x)
    y = x
    min = np.min(y)
    max = np.max(y)
    y = (y - min)/ (max - min)
    
    return np.abs(y)

def abs_torch(x):
    y = torch.abs(x)
    y = (y - y.min())/ (y.max() - y.min())
    
    return torch.abs(y)


