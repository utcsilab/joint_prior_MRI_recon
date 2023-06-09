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
from sampling_funcs import marginal_ablation_sampler, StackedRandomGenerator, posterior_sampler
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
parser.add_argument('--num_steps', type=int, default=750)
parser.add_argument('--num_inner_steps', type=int, default=4)
parser.add_argument('--R', type=int, default=4)
parser.add_argument('--sample', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--contrast_recon', type=str, default='PD')
parser.add_argument('--net_arch', type=str, default='ddpmpp') 
parser.add_argument('--discretization', type=str, default='edm') # ['vp', 've', 'iddpm', 'edm']
parser.add_argument('--solver', type=str, default='euler') # ['euler', 'heun']
parser.add_argument('--schedule', type=str, default='vp') # ['vp', 've', 'linear']
parser.add_argument('--scaling', type=str, default='vp') # ['vp', 'none']

args   = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
#seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
device=torch.device('cuda')


# load sample 
data_file = '/home/blevac/cond_score_data/fastMRI_knee/sample%d_R%d.pt'%(args.sample,args.R)
contents = torch.load(data_file)
# print(contents.keys())
if args.contrast_recon == 'PD':
    s_maps          = torch.tensor(contents['maps_1']).cuda() # shape: [1,C,H,W]
    ksp           = torch.tensor(contents['ksp_1']).cuda()# shape: [1,C,H,W]
    mask          = torch.tensor(contents['mask_1']).cuda() # shape: [1,1,H,W]
    gt_img        = torch.tensor(contents['gt_img_1']).cuda() # shape [1,1,H,W]
    norm_c        = torch.tensor(contents['norm1']).cuda() # scalar
    class_idx = None

elif args.contrast_recon == 'PDFS':
    s_maps          = torch.tensor(contents['maps_2']).cuda() # shape: [1,C,H,W]
    ksp           = torch.tensor(contents['ksp_2']).cuda()# shape: [1,C,H,W]
    mask          = torch.tensor(contents['mask_2']).cuda() # shape: [1,1,H,W]
    gt_img        = torch.tensor(contents['gt_img_2']).cuda() # shape [1,1,H,W]
    norm_c        = torch.tensor(contents['norm2']).cuda() # scalar
    class_idx = None

print(s_maps.shape)
# print(ksp.shape)
# normalize
ksp = ksp/norm_c
gt_img = gt_img/norm_c

batch_size = 1

# results_dir = '/csiNAS2/slow/brett/conformal_samples/results/DPS_marginal/%s/net-%s_step-%d_lss-%.1e_sigmaMax-%.1e/R%d/sample%d/seed%d/'%(args.contrast_recon, args.net_arch, args.num_steps,  args.l_ss, args.sigma_max, args.R, args.sample, args.seed)

results_dir = './results_ISMRM/DPS_marginal/%s/net-%s_step-%d_lss-%.1e_sigmaMax-%.1e/R%d/sample%d/seed%d/'%(args.contrast_recon, args.net_arch, args.num_steps,  args.l_ss, args.sigma_max, args.R, args.sample, args.seed)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# load network
if args.contrast_recon=='PD':
    net_save = '/csiNAS2/slow/brett/edm_outputs/00021-PD_new-uncond-ddpmpp-edm-gpus3-batch15-fp32-PD/network-snapshot-001557.pkl'
elif args.contrast_recon=='PDFS':
    net_save = '/csiNAS2/slow/brett/edm_outputs/00023-PDFS_new-uncond-ddpmpp-edm-gpus3-batch15-fp32-PDFS/network-snapshot-001557.pkl'

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

# run inference
# image_recon = marginal_ablation_sampler(y_meas=ksp, s_maps=s_maps, mask=mask,l_ss=args.l_ss, 
#                                         net=net, latents=latents,
#                                         class_labels=class_labels, randn_like=rnd.randn_like, 
#                                         num_steps=args.num_steps, num_inner_steps=args.num_inner_steps, solver=args.solver, 
#                                         discretization=args.discretization, schedule=args.schedule,
#                                         scaling=args.scaling, gt_img = gt_img)

image_recon, img_stack = posterior_sampler(net=net, gt_img=gt_img, y=ksp, maps=s_maps, mask=mask, 
                                latents=latents, l_ss=args.l_ss, class_labels=None, 
                                randn_like=torch.randn_like, num_steps=args.num_steps, sigma_min=0.002, 
                                sigma_max=args.sigma_max, rho=7, S_churn=420, S_min=0, S_max=float('inf'), S_noise=args.S_noise)

cplx_recon = torch.view_as_complex(image_recon.permute(0,-2,-1,1).contiguous())[None] #shape: [1,1,H,W]


print('Sample %d, Seed %d, NRMSE: %.3f'%(args.sample,args.seed, nrmse(abs(gt_img), abs(cplx_recon)).item()))

dict = { 'gt_img': gt_img.cpu().numpy(),
        'recon':cplx_recon.cpu().numpy(),
        'img_stack': img_stack,
}

torch.save(dict, results_dir + '/checkpoint.pt')


