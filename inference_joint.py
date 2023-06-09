# boiler plate imports
import h5py
import numpy as np
import glob
import torch
from tqdm import tqdm
# import sigpy as sp
import matplotlib.pyplot as plt
import os
import sys
import argparse
import copy
from dotmap import DotMap
import tqdm
from utils import forward, adjoint, nrmse
from utils import fft
from sampling_funcs import marginal_ablation_sampler, StackedRandomGenerator, joint_posterior_sampler
import re
import click
import tqdm
import pickle
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
# dist.init()


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--l_ss', type=float, default=1)
parser.add_argument('--S_noise', type=float, default=1)
parser.add_argument('--sigma_max', type=float, default=1)
parser.add_argument('--num_steps', type=int, default=100)
parser.add_argument('--num_inner_steps', type=int, default=4)
parser.add_argument('--R', type=int, default=4)
parser.add_argument('--sample', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--net_arch', type=str, default='ddpmpp') 
parser.add_argument('--discretization', type=str, default='edm') # ['vp', 've', 'iddpm', 'edm']
parser.add_argument('--solver', type=str, default='euler') # ['euler', 'heun']
parser.add_argument('--schedule', type=str, default='vp') # ['vp', 've', 'linear']
parser.add_argument('--scaling', type=str, default='vp') # ['vp', 'none']

args   = parser.parse_args()

#seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
device=torch.device('cuda')


# load sample 
data_file = '/home/blevac/cond_score_data/fastMRI_knee/sample%d_R%d.pt'%(args.sample,args.R)
contents = torch.load(data_file)
# print(contents.keys())
s_maps_1         = torch.tensor(contents['maps_1']).cuda() # shape: [1,C,H,W]
ksp_1           = torch.tensor(contents['ksp_1']).cuda()# shape: [1,C,H,W]
mask_1          = torch.tensor(contents['mask_1']).cuda() # shape: [1,1,H,W]
# mask_1 = torch.ones_like(s_maps_1)
gt_img_1        = torch.tensor(contents['gt_img_1']).cuda() # shape [1,1,H,W]
norm_c_1        = torch.tensor(contents['norm1']).cuda() # scalar
# ksp_1 = forward(image=gt_img_1, maps=s_maps_1, mask = mask_1)

s_maps_2        = torch.tensor(contents['maps_2']).cuda() # shape: [1,C,H,W]
ksp_2           = torch.tensor(contents['ksp_2']).cuda()# shape: [1,C,H,W]
# mask_2          = torch.tensor(contents['mask_2']).cuda() # shape: [1,1,H,W]
mask_2 = torch.ones_like(s_maps_2)
gt_img_2        = torch.tensor(contents['gt_img_2']).cuda() # shape [1,1,H,W]
norm_c_2        = torch.tensor(contents['norm2']).cuda() # scalar
ksp_2 = forward(image = gt_img_2, maps = s_maps_2, mask = mask_2)
class_idx = 0


# print(ksp.shape)
# normalize
ksp_1 = ksp_1/norm_c_1
gt_img_1 = gt_img_1/norm_c_1

ksp_2 = ksp_2/norm_c_1
gt_img_2 = gt_img_2/norm_c_1

print(torch.max(abs(gt_img_1)))
print(torch.max(abs(gt_img_2)))


batch_size = 1

results_dir = './results/DPS_joint/Joint/net-%s_step-%d_lss-%.1e/R%d/sample%d/'%( args.net_arch, args.num_steps,  args.l_ss,args.R, args.sample)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# load network

net_save = '/csiNAS2/slow/brett/edm_outputs/00019-fastmri_joint_knee_preprocessed-cond-ddpmpp-edm-gpus3-batch15-fp32-ddpmpp_1/network-snapshot-001909.pkl'

if dist.get_rank() != 0:
        torch.distributed.barrier()

# Load network.
dist.print0(f'Loading network from "{net_save}"...')
with dnnlib.util.open_url(net_save, verbose=(dist.get_rank() == 0)) as f:
    net = pickle.load(f)['ema'].to(device)



# Pick latents and labels.
rnd = StackedRandomGenerator(device, [args.seed])
latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
class_labels = None
if net.label_dim:
    class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
if class_idx is not None:
    class_labels[:, :] = 0
    class_labels[:, class_idx] = 1


image_recon_1, image_recon_2 = joint_posterior_sampler(net=net, gt_img_1=gt_img_1, y_1=ksp_1, maps_1=s_maps_1, mask_1=mask_1, gt_img_2=gt_img_2, y_2=ksp_2, maps_2=s_maps_2, mask_2=mask_2, 
                                latents=latents, l_ss=args.l_ss, class_labels=class_labels, 
                                randn_like=torch.randn_like, num_steps=args.num_steps, sigma_min=0.002, 
                                sigma_max=args.sigma_max, rho=7, S_churn=420, S_min=0, S_max=float('inf'), S_noise=args.S_noise)

cplx_recon_1 = torch.view_as_complex(image_recon_1.permute(0,-2,-1,1).contiguous())[None] #shape: [1,1,H,W]
cplx_recon_2 = torch.view_as_complex(image_recon_2.permute(0,-2,-1,1).contiguous())[None] #shape: [1,1,H,W]


print('img1 NRMSE: %.3f'%(nrmse(abs(gt_img_1), abs(cplx_recon_1)).item()))
print('img2 NRMSE: %.3f'%(nrmse(abs(gt_img_2), abs(cplx_recon_2)).item()))

dict = { 'gt_img_1': gt_img_1.detach().cpu().numpy(),
        'recon_1':cplx_recon_1.detach().cpu().numpy(),
        'gt_img_2': gt_img_2.detach().cpu().numpy(),
        'recon_2':cplx_recon_2.detach().cpu().numpy()
}

torch.save(dict, results_dir + '/checkpoint.pt')


