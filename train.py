from json import encoder
from model.pdd_gan import PDDGAN_Model
from torch.nn.modules import loss
from pytorch_ssim import ssim
import os
import sys
import argparse
import numpy as np
import json
import torch
import logging
import scipy.io as sio
from thop import profile
from torch.utils.data.dataset import T
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


from utils.unaligned_dataset import IXI_dataset,fastmri_dataset
from utils.vis_tools import Visualizer
from utils.math import AverageMeter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MyProject')

    # model name
    parser.add_argument('--experiment_name', type=str, default='train_pdd_fmri_T1_random_4X', help='give a experiment name before training')
    parser.add_argument('--model_type', type=str, default='pdd', help='model type') # recut/ recycle
    parser.add_argument('--resume', type=str, default=None, help='Filename of the checkpoint to resume')

    # dataset
    parser.add_argument('--data_root', type=str, default='/home/harddisk8T/lbw/fast_mri_brain/', help='data root folder')

    # model architectures
    parser.add_argument('--net_G', type=str, default='Unet', help='generator network') 
    parser.add_argument('--net_D', type=str, default='normal', help='discriminator network')
    parser.add_argument('--input_nc', type=int, default=2, help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=2, help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--gan_mode', type=str, default='vanilla', help='the loss of gan')
    parser.add_argument('--init_type', type=str, default='xavier', choices=['normal', 'xavier', 'kaiming', 'orthogonal'], help='network initialization')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')  
    parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images. if nce ,the pool size = 0 ')

    # training options
    parser.add_argument('--batch_size', type=int, default=6, help='training batch size')
    parser.add_argument('--mask_path', type=str, default='./mask_fmri/random_256_256_4X.mat', help='the path of mask')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=50, help='number of epochs to linearly decay learning rate to zero')
    # evaluation options
    parser.add_argument('--eval_epochs', type=int, default=4, help='evaluation epochs')
    parser.add_argument('--save_epochs', type=int, default=4, help='save evaluation for every number of epochs')

    # optimizer
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')

    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for ADAM')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for ADAM')

    # learning rate policy
    parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate decay policy')
    parser.add_argument('--lr_decay_iters', type=int, default=4, help='multiply by a gamma every lr_decay_iters iterations')

    # logger options
    parser.add_argument('--vis_freq', type=int, default=240, help='save model for every number of epochs')
    parser.add_argument('--output_path', default='./', type=str, help='Output path.')
    parser.add_argument('--save_freq', type=int, default=5, help='save model for every number of epochs')

    # other
    parser.add_argument('--num_workers', type=int, default=8, help='number of threads to load data')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='list of gpu ids')
    opts = parser.parse_args()

    options_str = json.dumps(opts.__dict__, indent=4, sort_keys=False)
    print("------------------- Options -------------------")
    print(options_str[2:-2])
    print("-----------------------------------------------")

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s',
                            filename=opts.experiment_name, filemode='a')

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # device-ids=0,1

    mask = sio.loadmat(opts.mask_path)['Umask']
    mask_under = np.stack((mask, mask), axis=-1)
    mask_under = torch.from_numpy(mask_under).float()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mask_torch = torch.from_numpy(mask).float().cuda()
    mask_norm = np.stack((mask, mask), axis=-1)
    mask_norm = torch.from_numpy(mask_norm).type(torch.uint8).cuda().unsqueeze(0)

    # initialize dataset
    train_dataset = fastmri_dataset(opts, mask_under, mask_under, mask_under, mode='TRAIN')
    val_dataset = fastmri_dataset(opts, mask_under, mask_under, mask_under, mode='VAL')

    train_loader = DataLoader(dataset=train_dataset, num_workers=opts.num_workers, batch_size=opts.batch_size,shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, num_workers=opts.num_workers, batch_size=opts.batch_size,shuffle=True, pin_memory=True, drop_last=True)

    # prepare output directories
    output_directory = os.path.join(opts.output_path,'outputs_'+ opts.experiment_name)
    os.makedirs(output_directory, exist_ok=True)
    image_directory = os.path.join(output_directory, 'images')
    os.makedirs(image_directory, exist_ok=True)
    checkpoints_directory = os.path.join(output_directory, 'checkpoints')
    os.makedirs(checkpoints_directory, exist_ok=True)


    if opts.model_type == 'pdd':
        model = PDDGAN_Model(mask_torch, opts, mask_under)
    

    vis_train = Visualizer(env='train')
    vis_validation = Visualizer(env='validation')

    sum_avg_psnr = AverageMeter()
    sum_avg_ssim = AverageMeter()
    sum_avg_loss = AverageMeter()
    best_psnr = 0


        

    start_epoch = 0
    print('Start training at epoch {} \n'.format(start_epoch))

    

    try: #Catch keyboard interrupt and save state
        for epoch in range(start_epoch, opts.n_epochs + opts.n_epochs_decay + 1):
            
            model.train()
            progress = 0
            epoch_iter = 0

            with tqdm(desc=f'Epoch{epoch + 1}/{opts.n_epochs + 1}', unit='imgs') as pbar:
                #Train loop
                for batch in train_loader:
                    current_batch_size = batch['under_k_A'].shape[0]

                    if epoch == start_epoch and progress == 0: 
                        model.set_scheduler(opts, -1)
                        if opts.resume is not None:
                            model.resume(opts.resume)
                    model.set_input(batch)
                    model.optimize_parameters()

                    loss_print = model.loss_summary
                    progress += 100*current_batch_size/len(train_dataset)
                    epoch_iter += current_batch_size
                    if epoch_iter % opts.vis_freq == 0:
                        counter_ratio = epoch_iter / len(train_dataset)
                        model.visualize_visdom(vis_train, epoch, counter_ratio)
                    pbar.set_postfix(**{'loss_print': loss_print,'Prctg of train set':progress})
                    pbar.update(current_batch_size)
            
            # On epoch end
            model.visualize_visdom(vis_train, epoch)
            model.save_results(image_directory, epoch)

            

            # Validation
            model.eval()
            with torch.no_grad():
                psnr_recon, ssim_recon, loss_recon = model.evaluate(val_loader, epoch)
                sum_avg_psnr.update(psnr_recon)
                sum_avg_ssim.update(ssim_recon)
                sum_avg_loss.update(loss_recon)

                logging.info(f'epoch:{epoch} psnr_avg:{sum_avg_psnr.avg} ssim_avg:{sum_avg_ssim.avg} loss:{sum_avg_loss.avg}')

            model.update_learning_rate()
            # visualization
            model.visualize_visdom(vis_validation, epoch, train=False, val=True)

            #Save Checkpoint
            if psnr_recon > best_psnr or epoch == 0:
                checkpoints_name = os.path.join(checkpoints_directory, f'CP_epoch{epoch+1}.pth')
                model.save(checkpoints_name, epoch)
                logging.info(f'Checkpoint {epoch+1} saved !')
                best_psnr = psnr_recon

    except KeyboardInterrupt:
        checkpoints_name = os.path.join(checkpoints_directory, f'CP_epoch{epoch}_INTERRUPTED.pth')
        model.save(checkpoints_name, epoch)
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

