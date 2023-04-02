import os
from numpy import dtype
import torch
from torch import nn
from torchvision.utils import save_image
from tqdm import tqdm
from math import log10
import logging
import pytorch_ssim
from utils.math import complex_abs_compare, fft,ifft,AverageMeter, vis_img, vis_kspace, complex_abs_eval, complex_abs
from networks.networks import DC_I, DC, Fusion, get_scheduler
from networks.generators import FeatureForwardUnit
from networks import get_generator, get_discriminator, get_MLP
from networks import networks
# from utils.select_mask import create_mask
from utils.image_pool import ImagePool
import itertools
import numpy as np
from skimage.metrics import structural_similarity as ssim
from utils.select_mask import create_mask, fast_mri_mask

def b_nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    batch_nmse = np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2
    return batch_nmse

def b_psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    mse = ((gt - pred) ** 2).mean()
    max_i = gt.max()
    s_psnr = 10 * np.log10((max_i ** 2) / mse)
    return s_psnr

def b_ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    s_ssim = ssim(gt, pred)
    
    return s_ssim

class FeatureExtractor(nn.Module):
    def __init__(self, opts, attention=False):
        super(FeatureExtractor, self).__init__()

        self.kspace_extractor = get_generator('simpleconv', opts.input_nc, opts.input_nc, opts.ngf, opts.init_type, opts.init_gain, opts.gpu_ids, opts, kspace = True, attention=attention)
        self.image_extractor = get_generator('simpleconv', opts.input_nc, opts.output_nc, opts.ngf, opts.init_type, opts.init_gain, opts.gpu_ids, opts, attention=attention)

    def forward(self, *input):
        k, img = input
        k_feature = self.kspace_extractor(k)
        img_feature = self.image_extractor(img)

        return k_feature, img_feature

class MRIReconstruction(nn.Module):
    def __init__(self, mask_img, mask_kspace, opts):
        super(MRIReconstruction, self).__init__()
        self.block1 = MRIRecBlock(mask_img, mask_kspace, opts)
        self.block2 = MRIRecBlock(mask_img, mask_kspace, opts)
        self.block3 = MRIRecBlock(mask_img, mask_kspace, opts)
        self.block4 = MRIRecBlock(mask_img, mask_kspace, opts)
        self.block5 = MRIRecBlock(mask_img, mask_kspace, opts)
        self.block6 = MRIRecBlock(mask_img, mask_kspace, opts)
        self.block7 = MRIRecBlock(mask_img, mask_kspace, opts)

        self.mask_img = mask_img
        self.mask_kspace = mask_kspace

        self.lambda_loss = nn.Parameter(torch.tensor(100.0, dtype=torch.float), requires_grad=True)
        
        

    def forward(self, *input):
        
        k_x_1 = input[1]
        img_x_1 = input[0]
        mask = input[2]

        u_k = k_x_1
        u_img = img_x_1

        k_x_2, img_x_2 = self.block1(k_x_1, img_x_1, mask, u_k)  

        k_x_3, img_x_3 = self.block2(k_x_2, img_x_2, mask, u_k) 

        k_x_4, img_x_4 = self.block3(k_x_3, img_x_3, mask, u_k) 

        k_x_5, img_x_5 = self.block4(k_x_4, img_x_4, mask, u_k) 

        k_x_6, img_x_6 = self.block5(k_x_5, img_x_5, mask, u_k) 

        k_x_7, img_x_7 = self.block6(k_x_6, img_x_6, mask, u_k)


        out_k, out_img = self.block7(k_x_7, img_x_7, mask, u_k) 

        return  img_x_2, k_x_2, img_x_3, k_x_3, img_x_4, k_x_4, \
                img_x_5, k_x_5, img_x_6, k_x_6, img_x_7, k_x_7, \
                out_img, out_k
        

class MRIRecBlock(nn.Module):
    def __init__(self, mask_img, mask_kspace, opts):
        super(MRIRecBlock, self).__init__()
        self.cnn = FeatureExtractor(opts, attention=True)
        self.dc1 = DC()
        self.dc2 = DC()
        self.fusion1 = Fusion()
        self.fusion2 = Fusion()
        
    
    def forward(self, k_x, img_x, mask,u_k):


        k_fea, img_fea = self.cnn(*(k_x, img_x))

        rec_k = self.dc1(k_fea, u_k, mask)
        rec_img = self.dc2(img_fea, u_k, mask, True)

        k_to_img = ifft(rec_k)  # convert the restored kspace to spatial domain
        img_to_k = fft(rec_img) # convert the restored image to frequency domain

        k_x_2 = self.fusion1(rec_k, img_to_k)
        img_x_2 = self.fusion2(rec_img, k_to_img)

        return k_x_2, img_x_2

class PDDGAN_Model(nn.Module):
    def __init__(self, mask, opts, u_mask):
        super(PDDGAN_Model, self).__init__()
        self.mask = mask
        self.mask_under = u_mask.cuda()
        self.opts = opts
        self.optimizers = []
        self.netG_A = MRIReconstruction(self.mask, self.mask, self.opts)
        self.device = torch.device('cuda:{}'.format(opts.gpu_ids[0])) if opts.gpu_ids else torch.device('cpu')
        self.criterionGAN = networks.GANLoss(opts.gan_mode).to(self.device)
        self.criterionCycleL1 = nn.L1Loss()
        self.criterionIdt = nn.L1Loss()
        self.criterionCycleK = nn.L1Loss(reduction='sum')
        self.criterionIdtK = nn.L1Loss(reduction='sum')
        self.lambda_idt = 0.5
        self.lambda_A = 10.0
        self.lambda_B = 10.0
        self.lambda_adv = 1.0
        self.netD_I_A = get_discriminator(opts.net_D, opts.input_nc, opts.ndf, opts.init_type, opts.init_gain, opts.gpu_ids, opts)
        self.netD_K_A = get_discriminator(opts.net_D, opts.input_nc, opts.ndf, opts.init_type, opts.init_gain, opts.gpu_ids, opts, kspace = True)
        self.fake_I_A_pool = ImagePool(opts.pool_size)  # create image buffer to store previously generated images
        self.fake_I_B_pool = ImagePool(opts.pool_size)  # create image buffer to store previously generated images
        self.fake_K_A_pool = ImagePool(opts.pool_size)  # create image buffer to store previously generated images
        self.fake_K_B_pool = ImagePool(opts.pool_size)  # create image buffer to store previously generated images
        self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(), lr=2e-4, betas=(opts.beta1, 0.999))
        self.optimizer_D_I = torch.optim.Adam(self.netD_I_A.parameters(), lr=1e-5, betas=(opts.beta1, 0.999))
        self.optimizer_D_K = torch.optim.Adam(self.netD_K_A.parameters(), lr=1e-6, betas=(opts.beta1, 0.999))
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D_I)
        self.optimizers.append(self.optimizer_D_K)
        self.lambda_loss = self.netG_A.lambda_loss
        self.create_mask = create_mask(ratio=0.33)


    def cycle_forward(self, real_B, fake_B, kspace=False):
        if kspace:
            rec_A = self.inverse_operator(fake_B, kspace=True)
            fake_A = self.inverse_operator(real_B, down=True, kspace=True)
        else:
            rec_A = self.inverse_operator(fake_B)
            fake_A = self.inverse_operator(real_B, down=True)
            
        return rec_A, fake_A
    
    def inverse_operator(self, x, down=False, kspace=False):
        if down:
            c_mask = self.create_mask.uniform_creation(shape=(256,256), vertical=True)
            self.c_mask = np.stack((c_mask, c_mask), axis=-1)
            mask = torch.from_numpy(self.c_mask).float().cuda().permute(2,0,1) 
            mask = self.mask.cuda()

            index = int(np.random.rand() / 0.2)
            c_mask = self.mask_random_radial[index]
            # # c_mask = self.mask_random_radial
            self.c_mask = np.stack((c_mask, c_mask), axis=-1)
            mask = torch.from_numpy(self.c_mask).float().cuda().permute(2,0,1)

            # c_mask = self.mask
            # mask = self.mask
        else:
            mask = self.mask
        if not kspace:
            x = fft(x)
        x2 = mask * x
        if kspace:
            return x2
        else:
            x3 = ifft(x2)
            return x3
    

    def forward(self):

        self.real_A_i = self.under_img_A
        self.real_B_i = self.full_img_B
        self.real_A_k = self.under_k_A
        self.real_B_k = self.full_k_B
        
        # fake_A, fake_B
        
        self.fake_B_i1, self.fake_B_k1, self.fake_B_i2, self.fake_B_k2, self.fake_B_i3, self.fake_B_k3,\
        self.fake_B_i4, self.fake_B_k4, self.fake_B_i5, self.fake_B_k5, self.fake_B_i6, self.fake_B_k6,\
        self.fake_B_i7, self.fake_B_k7 = self.netG_A(*(self.under_img_A,self.under_k_A, self.mask_under))


        self.rec_A_i1, self.fake_A_i1 = self.cycle_forward(self.real_B_i, self.fake_B_i1)
        self.rec_A_k1, self.fake_A_k1 = self.cycle_forward(self.real_B_k, self.fake_B_k1, kspace=True)
        self.rec_B_k1, self.rec_B_i1 = self.netG_A.block1(self.fake_A_k1, self.fake_A_i1, self.mask_under, self.under_k_A)
        self.idt_A_k1, self.idt_A_i1 = self.netG_A.block1(self.real_B_k, self.real_B_i, self.mask_under, self.under_k_A)

        
        self.rec_A_i2, self.fake_A_i2 = self.cycle_forward(self.real_B_i, self.fake_B_i2)
        self.rec_A_k2, self.fake_A_k2 = self.cycle_forward(self.real_B_k, self.fake_B_k2, kspace=True)
        self.rec_B_k2, self.rec_B_i2 = self.netG_A.block2(self.fake_A_k2, self.fake_A_i2, self.mask_under, self.under_k_A)
        self.idt_A_k2, self.idt_A_i2 = self.netG_A.block2(self.real_B_k, self.real_B_i, self.mask_under, self.under_k_A)
       

        self.rec_A_i3, self.fake_A_i3 = self.cycle_forward(self.real_B_i, self.fake_B_i3)
        self.rec_A_k3, self.fake_A_k3 = self.cycle_forward(self.real_B_k, self.fake_B_k3, kspace=True)
        self.rec_B_k3, self.rec_B_i3 = self.netG_A.block3(self.fake_A_k3, self.fake_A_i3, self.mask_under, self.under_k_A)
        self.idt_A_k3, self.idt_A_i3 = self.netG_A.block3(self.real_B_k, self.real_B_i, self.mask_under, self.under_k_A)

        self.rec_A_i4, self.fake_A_i4 = self.cycle_forward(self.real_B_i, self.fake_B_i4)
        self.rec_A_k4, self.fake_A_k4 = self.cycle_forward(self.real_B_k, self.fake_B_k4, kspace=True)
        self.rec_B_k4, self.rec_B_i4 = self.netG_A.block4(self.fake_A_k4, self.fake_A_i4, self.mask_under, self.under_k_A)
        self.idt_A_k4, self.idt_A_i4 = self.netG_A.block4(self.real_B_k, self.real_B_i, self.mask_under, self.under_k_A)

        self.rec_A_i5, self.fake_A_i5 = self.cycle_forward(self.real_B_i, self.fake_B_i5)
        self.rec_A_k5, self.fake_A_k5 = self.cycle_forward(self.real_B_k, self.fake_B_k5, kspace=True)
        self.rec_B_k5, self.rec_B_i5 = self.netG_A.block5(self.fake_A_k5, self.fake_A_i5, self.mask_under, self.under_k_A)
        self.idt_A_k5, self.idt_A_i5 = self.netG_A.block5(self.real_B_k, self.real_B_i, self.mask_under, self.under_k_A)

        self.rec_A_i6, self.fake_A_i6 = self.cycle_forward(self.real_B_i, self.fake_B_i6)
        self.rec_A_k6, self.fake_A_k6 = self.cycle_forward(self.real_B_k, self.fake_B_k6, kspace=True)
        self.rec_B_k6, self.rec_B_i6 = self.netG_A.block6(self.fake_A_k6, self.fake_A_i6, self.mask_under, self.under_k_A)
        self.idt_A_k6, self.idt_A_i6 = self.netG_A.block6(self.real_B_k, self.real_B_i, self.mask_under, self.under_k_A)

        self.rec_A_i7, self.fake_A_i7 = self.cycle_forward(self.real_B_i, self.fake_B_i7)
        self.rec_A_k7, self.fake_A_k7 = self.cycle_forward(self.real_B_k, self.fake_B_k7, kspace=True)
        self.rec_B_k7, self.rec_B_i7 = self.netG_A.block7(self.fake_A_k7, self.fake_A_i7, self.mask_under, self.under_k_A)
        self.idt_A_k7, self.idt_A_i7 = self.netG_A.block7(self.real_B_k, self.real_B_i, self.mask_under, self.under_k_A)


        self.fake_B_i = self.fake_B_i7

        
        return self.fake_B_i
    
    
    def forward_test(self):

        _, _, _, _, _, _, _, _, _, _, _, _, self.fake_B_i, self.fake_B_k = self.netG_A(*(self.under_img_A,self.under_k_A, self.mask_under))
        
        return self.fake_B_i, self.label

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """

        pred_real = netD(real.detach())
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5
        # loss_D.backward()
        return loss_D

    def calculate_loss_D_A(self, fake_B, real_B, netD_A, fake_B_pool):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = fake_B_pool.query(fake_B)
        # fake_B =self.fake_B
        self.loss_D_A = self.backward_D_basic(netD_A, real_B, fake_B)
        return self.loss_D_A 

    def calculate_loss_D_B(self, fake_A, real_A, netD_B, fake_A_pool):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = fake_A_pool.query(fake_A)
        # fake_A =self.fake_A
        self.loss_D_B = self.backward_D_basic(netD_B, real_A, fake_A)
        return self.loss_D_B

    def calculate_loss_G(self,idt_A, real_A, real_B, fake_B, rec_A, rec_B, netD_A, kspace=False):
        """Calculate the loss for generators G_A"""
        # Identity loss
        # G_A should be identity if real_B is fed: ||G_A(B) - B||
        loss_idt_A = self.criterionIdt(idt_A, real_B) * self.lambda_B * self.lambda_idt
        # GAN loss D_A(G_A(A))
        loss_G_A = self.criterionGAN(netD_A(fake_B), True) 
        # Forward cycle loss || G_B(G_A(A)) - A||
        loss_cycle_A = self.criterionCycleL1(rec_A, real_A) * self.lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        loss_cycle_B = self.criterionCycleL1(rec_B, real_B) * self.lambda_B 
            
        loss_G = loss_G_A + loss_cycle_A + loss_cycle_B + loss_idt_A
        
        return loss_G_A
    

        
        
    def set_input(self, data):
        # label for validition or test
        self.label = data['target_img_A'].type(torch.FloatTensor).to(self.device)
        self.label_k = fft(self.label)
        # image space
        self.under_img_A = data['under_img_A'].type(torch.FloatTensor).to(self.device)
        self.full_img_B = data['full_img_B'].type(torch.FloatTensor).to(self.device)
        
        # kspace
        self.under_k_A = data['under_k_A'].type(torch.FloatTensor).to(self.device)
        self.full_k_B = fft(self.full_img_B)

    
    def set_input_test(self,under_k, under_img, target_img):
        self.under_k_A = under_k.to(self.device, dtype=torch.float)
        self.under_img_A = under_img.to(self.device, dtype=torch.float)
        self.label = target_img.to(self.device,dtype=torch.float)
    
    def calculate_G_loss(self):
        loss_G_I1 = self.calculate_loss_G(self.idt_A_i1, self.real_A_i, self.real_B_i, self.fake_B_i1, self.rec_A_i1, self.rec_B_i1, self.netD_I_A)
        loss_G_I2 = self.calculate_loss_G(self.idt_A_i2, self.real_A_i, self.real_B_i, self.fake_B_i2, self.rec_A_i2, self.rec_B_i2, self.netD_I_A)
        loss_G_I3 = self.calculate_loss_G(self.idt_A_i3, self.real_A_i, self.real_B_i, self.fake_B_i3, self.rec_A_i3, self.rec_B_i3, self.netD_I_A)
        loss_G_I4 = self.calculate_loss_G(self.idt_A_i4, self.real_A_i, self.real_B_i, self.fake_B_i4, self.rec_A_i4, self.rec_B_i4, self.netD_I_A)
        loss_G_I5 = self.calculate_loss_G(self.idt_A_i5, self.real_A_i, self.real_B_i, self.fake_B_i5, self.rec_A_i5, self.rec_B_i5, self.netD_I_A)
        loss_G_I6 = self.calculate_loss_G(self.idt_A_i6, self.real_A_i, self.real_B_i, self.fake_B_i6, self.rec_A_i6, self.rec_B_i6, self.netD_I_A)
        loss_G_I7 = self.calculate_loss_G(self.idt_A_i7, self.real_A_i, self.real_B_i, self.fake_B_i7, self.rec_A_i7, self.rec_B_i7, self.netD_I_A)
       
        self.loss_G_I = loss_G_I1+ loss_G_I3+ loss_G_I5+ loss_G_I4 +loss_G_I2 + loss_G_I6 + loss_G_I7
        
        
        loss_G_K1 = self.calculate_loss_G(self.idt_A_k1, self.real_A_k, self.real_B_k, self.fake_B_k1, self.rec_A_k1, self.rec_B_k1, self.netD_K_A, kspace=True)
        loss_G_K2 = self.calculate_loss_G(self.idt_A_k2, self.real_A_k, self.real_B_k, self.fake_B_k2, self.rec_A_k2, self.rec_B_k2, self.netD_K_A, kspace=True)
        loss_G_K3 = self.calculate_loss_G(self.idt_A_k3, self.real_A_k, self.real_B_k, self.fake_B_k3, self.rec_A_k3, self.rec_B_k3, self.netD_K_A, kspace=True)
        loss_G_K4 = self.calculate_loss_G(self.idt_A_k4, self.real_A_k, self.real_B_k, self.fake_B_k4, self.rec_A_k4, self.rec_B_k4, self.netD_K_A, kspace=True)
        loss_G_K5 = self.calculate_loss_G(self.idt_A_k5, self.real_A_k, self.real_B_k, self.fake_B_k5, self.rec_A_k5, self.rec_B_k5, self.netD_K_A, kspace=True)
        loss_G_K6 = self.calculate_loss_G(self.idt_A_k6, self.real_A_k, self.real_B_k, self.fake_B_k6, self.rec_A_k6, self.rec_B_k6, self.netD_K_A, kspace=True)
        loss_G_K7 = self.calculate_loss_G(self.idt_A_k7, self.real_A_k, self.real_B_k, self.fake_B_k7, self.rec_A_k7, self.rec_B_k7, self.netD_K_A, kspace=True)
       
        self.loss_G_K = loss_G_K1  + loss_G_K3  + loss_G_K5 + loss_G_K4 +loss_G_K2 + loss_G_K6 + loss_G_K7 
        
        self.loss_G = self.loss_G_I + self.loss_G_K
        
    
    def calculate_D_I_loss(self):
        loss_D_I_A_1 = self.calculate_loss_D_A(self.fake_B_i1, self.real_B_i, self.netD_I_A, self.fake_I_B_pool)
        loss_D_I_A_2 = self.calculate_loss_D_A(self.fake_B_i2, self.real_B_i, self.netD_I_A, self.fake_I_B_pool)
        loss_D_I_A_3 = self.calculate_loss_D_A(self.fake_B_i3, self.real_B_i, self.netD_I_A, self.fake_I_B_pool)
        loss_D_I_A_4 = self.calculate_loss_D_A(self.fake_B_i4, self.real_B_i, self.netD_I_A, self.fake_I_B_pool)
        loss_D_I_A_5 = self.calculate_loss_D_A(self.fake_B_i5, self.real_B_i, self.netD_I_A, self.fake_I_B_pool)
        loss_D_I_A_6 = self.calculate_loss_D_A(self.fake_B_i6, self.real_B_i, self.netD_I_A, self.fake_I_B_pool)
        loss_D_I_A_7 = self.calculate_loss_D_A(self.fake_B_i7, self.real_B_i, self.netD_I_A, self.fake_I_B_pool)

        self.loss_D_I_A =loss_D_I_A_1 + loss_D_I_A_3 + loss_D_I_A_5 + loss_D_I_A_4 + loss_D_I_A_2 + loss_D_I_A_6 + loss_D_I_A_7 
    
    def calculate_D_k_loss(self):
        loss_D_k_A_1 = self.calculate_loss_D_A(self.fake_B_k1, self.real_B_k, self.netD_K_A, self.fake_K_B_pool)
        loss_D_k_A_2 = self.calculate_loss_D_A(self.fake_B_k2, self.real_B_k, self.netD_K_A, self.fake_K_B_pool)
        loss_D_k_A_3 = self.calculate_loss_D_A(self.fake_B_k3, self.real_B_k, self.netD_K_A, self.fake_K_B_pool)
        loss_D_k_A_4 = self.calculate_loss_D_A(self.fake_B_k4, self.real_B_k, self.netD_K_A, self.fake_K_B_pool)
        loss_D_k_A_5 = self.calculate_loss_D_A(self.fake_B_k5, self.real_B_k, self.netD_K_A, self.fake_K_B_pool)
        loss_D_k_A_6 = self.calculate_loss_D_A(self.fake_B_k6, self.real_B_k, self.netD_K_A, self.fake_K_B_pool)
        loss_D_k_A_7 = self.calculate_loss_D_A(self.fake_B_k7, self.real_B_k, self.netD_K_A, self.fake_K_B_pool)
        
        self.loss_D_k_A = loss_D_k_A_1  +loss_D_k_A_3 + loss_D_k_A_5 +loss_D_k_A_4 +loss_D_k_A_2 + loss_D_k_A_6 + loss_D_k_A_7 
        
        
    

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A
        self.set_requires_grad(self.netD_I_A, False)  # Ds require no gradients when optimizing Gs
        self.set_requires_grad(self.netD_K_A, False)
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.calculate_G_loss()             # calculate gradients for G_A and G_B
        self.loss_G.backward()
        self.running_G_loss.update(self.loss_G)
        
        # D_A
        self.set_requires_grad(self.netD_I_A, True)
        self.set_requires_grad(self.netD_K_A, True)
        self.optimizer_D_I.zero_grad()   # set D_A and D_B's gradients to zero
        self.optimizer_D_K.zero_grad()
        self.calculate_D_I_loss()      # calculate gradients for D_A
        self.calculate_D_k_loss()      # calculate graidents for D_B
        self.loss_D = self.loss_D_I_A + 0.001*self.loss_D_k_A
        self.running_D_loss.update(self.loss_D)
        
    
        self.loss_D_I_A.backward()
        self.loss_D_k_A.backward()
        self.optimizer_G.step()       # update G_A and G_B's weights
        self.optimizer_D_I.step()
        self.optimizer_D_K.step()
        
    def save(self, filename, epoch):
        state = {}
        
        state['netG'] = self.netG_A.state_dict()
        state['netD_I_A'] = self.netD_I_A.state_dict()
        state['netD_k_A'] = self.netD_K_A.state_dict()

        state['opt_G'] = self.optimizer_G.state_dict()
        state['opt_D_I'] = self.optimizer_D_I.state_dict()
        state['opt_D_k'] = self.optimizer_D_K.state_dict()
        state['epoch'] = epoch

        torch.save(state, filename)
        print('Saved {}'.format(filename))

    def resume(self, checkpoint_file, train=True):
        checkpoint = torch.load(checkpoint_file)
        self.netG_A.load_state_dict(checkpoint['netG'])
        self.netD_I_A.load_state_dict(checkpoint['netD_I_A'])
        self.netD_K_A.load_state_dict(checkpoint['netD_k_A'])

        self.optimizer_G.load_state_dict(checkpoint['opt_G'])
        self.optimizer_D_I.load_state_dict(checkpoint['opt_D_I'])
        self.optimizer_D_K.load_state_dict(checkpoint['opt_D_k'])

        print('Loaded {}'.format(checkpoint_file))

        return checkpoint['epoch']

    def resume_epoch(self, checkpoint_file, train=True):
        checkpoint = torch.load(checkpoint_file)
        return checkpoint['epoch']

    @property
    def loss_summary(self):
        message = ''
        
        message += 'G_I_loss: {:.4e} D_I_A_loss: {:.4e} D_K_A_loss: {:.4e}'\
            .format(self.loss_G, self.loss_D_I_A, self.loss_D_k_A)

        return message

    def evaluate(self, loader, epoch):
        val_bar = tqdm(loader)
        avg_psnr = AverageMeter()
        avg_ssim = AverageMeter()
        avg_nmse = AverageMeter()

        for data in val_bar:
            self.set_input(data)
            self.forward_test()
            for i in range(self.opts.batch_size):
                self.out_val = complex_abs_eval(self.fake_B_i[i:i+1,...])
                self.label_val = complex_abs_eval(self.label[i:i+1,...])

                ssim_recon = b_ssim(self.label_val.squeeze().cpu().numpy(), self.out_val.squeeze().cpu().numpy())
                avg_ssim.update(ssim_recon)
            

                nmse_recon = b_nmse(self.label_val.squeeze().cpu().numpy(), self.out_val.squeeze().cpu().numpy())
                avg_nmse.update(nmse_recon)
                

                psnr_recon = b_psnr(self.label_val.squeeze().cpu().numpy(), self.out_val.squeeze().cpu().numpy())
                avg_psnr.update(psnr_recon)
            

            message = 'PSNR:{:4f}'.format(avg_psnr.avg)
            message += 'SSIM:{:4f}'.format(avg_ssim.avg)
            message += 'LOSS:{:4f}'.format(avg_nmse.avg)
            val_bar.set_description(desc = message)

        self.psnr_recon = avg_psnr.avg
        self.ssim_recon = avg_ssim.avg
        self.loss_recon = avg_nmse.avg
        

        logging.info(f'epoch:{epoch} psnr:{self.psnr_recon} ssim:{self.ssim_recon} loss:{self.loss_recon}')

        return self.psnr_recon, self.ssim_recon, self.loss_recon


    def visualize_visdom(self, vis, epoch, counter_ratio=0, train=True, val=True):
        if train:
            vis.plot("train / loss_G", self.running_G_loss.avg.item(), epoch+counter_ratio)
            vis.plot("train / loss_D", self.running_D_loss.avg.item(), epoch+counter_ratio)

            vis.img("train / u_img_A", vis_img(complex_abs_eval(self.under_img_A)))
            vis.img("train / full_img_B " , vis_img(complex_abs_eval(self.full_img_B)))
            vis.img("train / full_img_A " , vis_img(complex_abs_eval(self.label)))
            vis.img("Mask_up", self.mask.cpu().numpy())
            vis.img("Mask2", self.c_mask[:,:,0])
            vis.img("train / Reconstructed_img_A ", vis_img(complex_abs_eval(self.fake_B_i)))
            
        elif val:
            vis.plot("validation / loss", self.loss_recon.item(), epoch)
            vis.plot("validation / psnr", self.psnr_recon, epoch)
            vis.plot("validation / ssim", self.ssim_recon.item(), epoch)
            
            vis.img("validation / u_img_A " , vis_img(complex_abs_eval(self.under_img_A)))
            vis.img("validation / full_img_B ", vis_img(complex_abs_eval(self.full_img_B)))
            vis.img("validation / full_img_A " , vis_img(complex_abs_eval(self.label)))
            vis.img("validation / Reconstructed_img_A ", vis_img(complex_abs_eval(self.fake_B_i)))
        
        

    def save_results(self, dir, epoch):
        pred = os.path.join(dir, 'pred_{:03d}.png'.format(epoch))
        gt = os.path.join(dir, 'gt_{:03d}.png'.format(epoch))
        input = os.path.join(dir, 'input_{:03d}.png'.format(epoch))

        fake_B =self.fake_B_i.detach()
        vis_recon = fake_B[0]
        save_image(vis_recon, pred, normalize=True, scale_each=True, padding=5)
        
        vis_gt = self.label[0]
        save_image(vis_gt, gt, normalize=True, scale_each=True, padding=5)
        
        real_A = self.under_img_A.detach()
        vis_input = real_A[0]
        save_image(vis_input, input, normalize=True, scale_each=True, padding=5)
    
    def set_scheduler(self, opts, epoch=-1):
        
        self.schedulers = [get_scheduler(optimizer, opts) for optimizer in self.optimizers]
    
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step(self.ssim_recon)
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = {:7f}'.format(lr))
    
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad