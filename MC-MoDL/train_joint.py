# boiler plate imports
import h5py
import numpy as np
import glob
import torch
from tqdm import tqdm
import sigpy as sp
import matplotlib.pyplot as plt
import os
import sys
import argparse
import copy
from dotmap import DotMap
import tqdm
from models import joint_MoDL
from torch.optim import Adam
from torch.utils.data import DataLoader
from datagen import fastMRI_knee

#seeds
torch.manual_seed(2022)
np.random.seed(2022)

def nrmse(x,y):
    num = torch.norm(x-y)
    den = torch.norm(x)
    return num/den


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--LR',  type=float, default=1e-4)
parser.add_argument('--R', type=int, default=4)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=5)
parser.add_argument('--save_interval', type=int, default=1)

args   = parser.parse_args()

# Always !!!
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False
torch.backends.cudnn.benchmark        = True
# set GPU
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


# configure model settings
hparams = DotMap()
hparams.verbose      = False
hparams.batch_size   = args.batch_size
hparams.max_cg_steps = 6
hparams.cg_eps       = 1e-6
hparams.unrolls      = 6

hparams.logging      = False
hparams.img_channels = 16
hparams.img_blocks   = 4
hparams.img_arch     = 'UNet'

hparams.l2lam_train = True
hparams.l2lam_init  = 0.1

# Create Model and Load weights
model = joint_MoDL(hparams).cuda()
model.train()


optimizer = Adam(model.parameters(), lr=args.LR)


# Count parameters
total_params = np.sum([np.prod(p.shape) for p
                        in model.parameters() if p.requires_grad])
print('Total parameters %d' % total_params)

train_files = glob.glob('/csiNAS2/slow/brett/fastMRI_knee_contrast_pairs/train/*.h5')
train_dataset = fastMRI_knee(train_files, center_slice=17, num_slices=20, ACS_perc=0.03, R=args.R)
train_loader  = DataLoader(train_dataset, batch_size=hparams.batch_size,
                            shuffle=True, num_workers=args.num_workers, drop_last=True)

# Training Logs
nrmse_log = []
running_training   = 0.0
running_training_1 = 0.0
running_training_2 = 0.0

results_dir = './results/R=%d/CG_steps%d_ch%d_pools%d_unrolls%d'%(args.R,hparams.max_cg_steps, hparams.img_channels,hparams.img_blocks, hparams.unrolls)

if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

for epoch_idx in range(args.epochs):
    for sample_idx, sample in enumerate(train_loader):
        # Move to CUDA
        for key in sample.keys():
            try:
                sample[key] = sample[key].cuda()
            except:
                pass

            # Get outputs
        img1_est, img2_est = model(sample, hparams.unrolls)

        loss1 = nrmse(abs(sample['gt_img_1']), abs(img1_est))
        loss2 = nrmse(abs(sample['gt_img_2']), abs(img2_est))
        loss = loss1+loss2

         # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            running_training = 0.99 * running_training + 0.01 * loss.item() if running_training > 0. else loss.item()
            running_training_1 = 0.99 * running_training_1 + 0.01 * loss1.item() if running_training_1 > 0. else loss1.item()
            running_training_2 = 0.99 * running_training_2 + 0.01 * loss2.item() if running_training_2 > 0. else loss2.item()

        # Verbose
        print('Epoch %d, Step %d, Total Batch loss %.3f,  Avg. NRMSE %.3f,  img1 loss %.3f,  Avg. img1 NRMSE %.3f,  img2 loss %.3f,  Avg. img2 NRMSE %.3f' % (
            epoch_idx,  sample_idx, loss.item(), running_training, loss1.item(), running_training_1, loss2, running_training_2))

    if epoch_idx%args.save_interval==0:
        torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                results_dir + '/ckpt_epoch' + str(epoch_idx) + '.pt')