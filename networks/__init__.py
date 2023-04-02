import torch
import torch.nn as nn
import numpy as np
from networks.generators import  UNet, FeatureForwardUnit, GatedFeatureForwardUnit
from networks.discriminators import NLayerDiscriminator, PatchGAN
from networks.networks import init_net


def set_gpu(network, gpu_ids):
    network.to(gpu_ids[0])
    return network


def get_generator(name, input_nc, output_nc, ngf, init_type='normal', init_gain=0.02, gpu_ids=[], opt=None, kspace=False, attention=False):

    if name == 'simpleconv':
        if kspace:
            network = GatedFeatureForwardUnit(attention=attention)
        else:
            network = FeatureForwardUnit(in_channels=input_nc, out_channels=output_nc, attention=attention)
    num_param = sum([p.numel() for p in network.parameters() if p.requires_grad])
    print('Number of generator parameters: %.3f M' % (num_param / 1e6))
    return init_net(network, init_type, init_gain, gpu_ids)

def get_discriminator(name, input_nc, ndf, init_type='normal', init_gain=0.02, gpu_ids=[], opt=None, kspace=False):
    if kspace:
        network = NLayerDiscriminator(input_nc, ndf, n_layers = 1)
    else:
        network = NLayerDiscriminator(input_nc, ndf)

    num_param = sum([p.numel() for p in network.parameters() if p.requires_grad])
    print('Number of discriminator parameters: %.3f M' % (num_param / 1e6))
    return init_net(network, init_type, init_gain, gpu_ids)
