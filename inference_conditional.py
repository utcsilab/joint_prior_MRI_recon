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
from utils import forward, adjoint, nrmse
import dnnlib
import pickle
from torch_utils import distributed as dist
from generate import edm_sampler, ablation_sampler, StackedRandomGenerator
dist.init()
#seeds
torch.manual_seed(2022)
np.random.seed(2022)



parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--noise_boost', type=float, default=1.0)
parser.add_argument('--noise_boost2', type=float, default=1.0)
parser.add_argument('--normalize_grad', type=int, default=1)
parser.add_argument('--dc_boost', type=float, default=1.)
parser.add_argument('--step_lr',  type=float, default=9e-7)
parser.add_argument('--skip_levels', type=int, default=0)
parser.add_argument('--level_steps', type=int, default=4)
parser.add_argument('--save_interval', type=int, default=100)
parser.add_argument('--R', type=int, default=4)
parser.add_argument('--sample', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--noise_lvl', type=float, default=0.01)
parser.add_argument('--contrast_recon', type=str, default='PD')


args   = parser.parse_args()

#seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Always !!!
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False
torch.backends.cudnn.benchmark        = True
# set GPU
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = 'cuda'



#Load Data


data_file = '/home/blevac/cond_score_data/fastMRI_knee/sample%d_R=%d.pt'%(args.noise_lvl,args.sample,args.R)
contents = torch.load(data_file)

if args.contrast_recon == 'PD':
    maps          = torch.tensor(contents['maps_1']).cuda() # shape: [1,C,H,W]
    ksp           = torch.tensor(contents['ksp_1']).cuda()# shape: [1,C,H,W]
    mask          = torch.tensor(contents['mask_1']).cuda() # shape: [1,1,H,W]
    gt_img        = torch.tensor(contents['gt_img_1']).cuda() # shape [1,1,H,W]
    norm_c        = torch.tensor(contents['norm1']).cuda() # scalar
    contrast2_img = torch.tensor(contents['gt_img_2']).cuda()/norm_c # shape [1,1,H,W]
    c1_idx = [0,1]
    c2_idx = [2,3]

elif args.contrast_recon == 'PDFS':
    maps          = torch.tensor(contents['maps_2']).cuda() # shape: [1,C,H,W]
    ksp           = torch.tensor(contents['ksp_2']).cuda()# shape: [1,C,H,W]
    mask          = torch.tensor(contents['mask_2']).cuda() # shape: [1,1,H,W]
    gt_img        = torch.tensor(contents['gt_img_2']).cuda() # shape [1,1,H,W]
    norm_c        = torch.tensor(contents['norm1']).cuda() # scalar
    contrast2_img = torch.tensor(contents['gt_img_1']).cuda()/norm_c # shape [1,1,H,W]
    c1_idx = [2,3]
    c2_idx = [0,1]



# logging
img_prog      = []
p_grad_prog   = []
img_nrmse_log = []

# Results Directory
results_dir = './results/conditional_val/contrast%s_noiseLVL%.1e_sample%d_R=%d_step_size%.2e_noiseSteps%d_dcBoost%.1e_skipLVLs%d_noiseBoost2%.2e'%(args.contrast_recon,args.noise_lvl,args.sample,args.R,args.step_lr, args.level_steps, args.dc_boost, args.skip_levels, args.noise_boost2)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)



# Create Model and Load weights

network_pkl = '/csiNAS2/slow/brett/edm_outputs/00003-fastmri_joint_knee_preprocessed-cond-ddpmpp-edm-gpus3-batch15-fp32-ddpmpp_1/network-snapshot-000904.pkl'
with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)


# Pick latents and labels.
rnd = StackedRandomGenerator(device, [args.seed])
#initialize samples
samples = torch.randn([1, net.img_channels, 320, 320], device=device)
class_labels = None
class_idx=0
if net.label_dim:
    class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[1], device=device)]
if class_idx is not None:
    class_labels[:, :] = 0
    class_labels[:, class_idx] = 1

    

contrast2_2ch = torch.view_as_real(contrast2_img[:,0,...]).permute(0,-1,-3,-2)
samples[:,c2_idx,...] = contrast2_2ch

with torch.no_grad():
    # run annealed langevin dynamics
    for noise_idx in tqdm(range(config.inference.num_steps)):
        # Get current noise power
        sigma  = score.sigmas[noise_idx + config.inference.sigma_offset]
        labels = torch.ones(samples.shape[0],device=samples.device) * (noise_idx + config.inference.sigma_offset)
        labels = labels.long()

        image_step_size = config.inference.step_lr * (sigma / score.sigmas[-1]) ** 2

        # for each noise level run several steps
        for step_idx in range(config.inference.num_steps_each):
            # Generate noise
            image_noise = torch.randn_like(samples) * torch.sqrt(args.noise_boost * image_step_size * 2)
            # samples[:,c2_idx,...] = contrast2_2ch + image_noise[:,c2_idx,...].float()
            samples[:,c2_idx,...] = contrast2_2ch + torch.randn_like(contrast2_2ch) * sigma*args.noise_boost2#torch.sqrt(args.noise_boost2*sigma * 2)
            # Get score model gradient
            p_grad = score(samples.float(), labels)

            # Convert contrast 1 part of samples to complex for DC Gradient Calc
            # print(samples[:,0:2,...].permute(0,-2,-1,1).shape)
            cplx_samples = norm_c*torch.view_as_complex(samples[:,c1_idx,...].permute(0,-2,-1,1).contiguous())[:,None,...]
            
            # Get Data Consistancy Gradient
            meas    = forward(image = cplx_samples, maps = maps, mask = mask) # shape: [B,C,H,W]
            # print('meas shape:',meas.shape)
            dc      = meas-ksp
            dc_loss = torch.norm(dc, p=2)**2

            if bool(args.normalize_grad):
                # normalize 
                # compute gradient, i.e., gradient = A_adjoint * ( y - Ax_hat )
                meas_grad = 2*torch.view_as_real(adjoint(ksp=dc, maps=maps, mask=mask)[:,0,...]).permute(0, 3, 1, 2)
                # meas_grad shape : [B,2,H,W]
                # Normalize, to make the gradient importance relatively the same
                meas_grad = meas_grad / torch.norm(meas_grad)
                meas_grad = meas_grad * torch.norm(p_grad[:,c1_idx,...])
                # meas_grad = meas_grad * torch.norm(p_grad)


    # dc_boost normalization is also occuring
            samples[:,c1_idx,...] = samples[:,c1_idx,...] + image_step_size * (p_grad[:,c1_idx,...] - args.dc_boost * meas_grad.float()) + image_noise[:,c1_idx,...].float()


            # Calculate metrics
            img_nrmse = nrmse(gt_img, cplx_samples)
            img_nrmse_log.append(img_nrmse)
            print('Step: %d,  sample: %d,  echo:%d,  NRMSE: %.4f'%(noise_idx, args.sample, args.echo_recon,img_nrmse.item()))

            # save periodically
            if noise_idx%args.save_interval == 0 and step_idx ==0:
                img_prog.append(cplx_samples)

                filename = results_dir + '/checkpoint.pt'

                torch.save({    'img_prog': img_prog,
                                'gt_img': gt_img.cpu(),
                                'img_nrmse':img_nrmse_log,
                                'args': args}, filename)



