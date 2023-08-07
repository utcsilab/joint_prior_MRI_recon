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
from tqdm import tqdm
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
parser.add_argument('--epoch', type=int, default=19)
parser.add_argument('--num_workers', type=int, default=5)

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
hparams.batch_size = 1
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

val_files = glob.glob('/csiNAS2/slow/brett/fastMRI_knee_contrast_pairs/val/*.h5')
val_dataset = fastMRI_knee(val_files, center_slice=17, num_slices=20, ACS_perc=0.03, R=args.R)
val_loader  = DataLoader(val_dataset, batch_size=1,
                            shuffle=True, num_workers=args.num_workers, drop_last=True)

# Training Logs
PD_nrmse = []
PDFS_nrmse = []
running_training   = 0.0
running_training_1 = 0.0
running_training_2 = 0.0

results_dir = './results/R=%d/CG_steps%d_ch%d_pools%d_unrolls%d'%(args.R,hparams.max_cg_steps, hparams.img_channels,hparams.img_blocks, hparams.unrolls)

model_file = results_dir + '/ckpt_epoch' + str(args.epoch) + '.pt'
cont = torch.load(model_file)
model.load_state_dict(cont['model_state_dict'])

model.eval()

for sample_idx, sample in tqdm(enumerate(val_loader)):
    # Move to CUDA
    for key in sample.keys():
        try:
            sample[key] = sample[key].cuda()
        except:
            pass

        # Get outputs
    img1_est, img2_est = model(sample, hparams.unrolls)

    PD_nrmse.append(nrmse(abs(sample['gt_img_1']), abs(img1_est)).item())
    PDFS_nrmse.append(nrmse(abs(sample['gt_img_2']), abs(img2_est)).item())

print('PD: ',np.mean(PD_nrmse),'   PDFS: ' ,np.mean(PDFS_nrmse))
torch.save({
        'PD_nrmse': PD_nrmse,
        'PDFS_nrmse': PDFS_nrmse},
        results_dir + '/val_results_R=' + str(args.R) + '.pt')
# save image of last example
