# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
from utils import forward, adjoint, nrmse


def posterior_sampler_vanilla(net, gt_img, y, maps, mask, latents, l_ss=1.0, second_order=False,
                     class_labels=None, randn_like=torch.randn_like, num_steps=100,
                      sigma_min=0.002, sigma_max=80, rho=7, S_churn=0,S_min=0,  S_max=float('inf'), S_noise=1, verbose=True):
    img_stack = []
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next
        x_hat = x_cur
        x_hat = x_hat.requires_grad_() #starting grad tracking with the noised img

        # Euler step.
        denoised = net(x_hat, t_cur, class_labels).to(torch.float64) # E[x_0|x_t]
        d_cur = (x_hat - denoised)/t_cur
        # take step over prior score and add noise
        x_next = x_hat + (t_next - t_cur) * d_cur #+ ((2*t_cur)*(t_cur-t_next))**0.5 * randn_like(x_cur)
        # print(((2*t_cur)*(t_cur-t_next))**0.5)
        # print(t_cur-t_next)
        # Likelihood step
        denoised_cplx = torch.view_as_complex(denoised.permute(0,-2,-1,1).contiguous())[None]

        Ax = forward(image=denoised_cplx, maps=maps, mask=mask)
        residual = y - Ax
        sse = torch.norm(residual)**2
        likelihood_score = torch.autograd.grad(outputs=sse, inputs=x_hat)[0]
        x_next = x_next - (l_ss / torch.sqrt(sse)) * likelihood_score

        # Cleanup 
        x_next = x_next.detach()
        x_hat = x_hat.requires_grad_(False)


        if verbose:
            cplx_recon = torch.view_as_complex(x_next.permute(0,-2,-1,1).contiguous())[None]
            print('Step:%d , NRMSE: %.3f'%(i, nrmse(gt_img, cplx_recon).item()))
            if i%10==0:
                img_stack.append(cplx_recon.cpu().numpy())

    return x_next, img_stack
  

def joint_posterior_sampler_vanilla(net, gt_img_1, y_1, maps_1, mask_1, gt_img_2, y_2, maps_2, mask_2, latents, l_ss=1.0, second_order=False,
                     class_labels=None, randn_like=torch.randn_like, num_steps=100,
                      sigma_min=0.002, sigma_max=80, rho=7, S_churn=0,S_min=0,  S_max=float('inf'), S_noise=1, verbose=True):

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

  
        # x_hat = x_hat #starting grad tracking with the noised img
        x_hat_1 = x_cur[:,0:2,...]
        x_hat_2 = x_cur[:,2:,...]
        x_hat_1 = x_hat_1.requires_grad_()
        x_hat_2 = x_hat_2.requires_grad_()
        x_hat_new = torch.cat((x_hat_1, x_hat_2), dim=1)
        # print(x_hat_new.shape)
        # Euler step.
        denoised = net(x_hat_new, t_cur, class_labels).to(torch.float64)
        d_cur = (x_hat_new - denoised) / t_cur
        x_next = x_hat_new + (t_next - t_cur) * d_cur

        # Likelihood step
        denoised_cplx_1 = torch.view_as_complex(denoised[:,0:2,...].permute(0,-2,-1,1).contiguous())[None]
        denoised_cplx_2 = torch.view_as_complex(denoised[:,2:,...].permute(0,-2,-1,1).contiguous())[None]

        Ax_1 = forward(image=denoised_cplx_1, maps=maps_1, mask=mask_1)
        Ax_2 = forward(image=denoised_cplx_2, maps=maps_2, mask=mask_2)
        residual_1 = y_1 - Ax_1
        residual_2 = y_2 - Ax_2
        sse_1 = torch.norm(residual_1)**2
        sse_2 = torch.norm(residual_2)**2



        likelihood_score_1 = torch.autograd.grad(outputs=sse_1, inputs=x_hat_1, retain_graph=True)[0]
        x_next[:,0:2,...] = x_next[:,0:2,...] - (l_ss / torch.sqrt(sse_1)) * likelihood_score_1
        likelihood_score_2 = torch.autograd.grad(outputs=sse_2, inputs=x_hat_2, retain_graph=True)[0]
        x_next[:,2:,...] = x_next[:,2:,...] - (l_ss / torch.sqrt(sse_2)) * likelihood_score_2

        x_next_1 = x_next[:,0:2,...]
        x_next_2 = x_next[:,2:,...]
        # Cleanup 
        x_next = x_next.detach()
        x_hat_1 = x_hat_1.requires_grad_(False)
        x_hat_2 = x_hat_2.requires_grad_(False)


        if verbose:
            cplx_recon_1 = torch.view_as_complex(x_next_1.permute(0,-2,-1,1).contiguous())[None]
            cplx_recon_2 = torch.view_as_complex(x_next_2.permute(0,-2,-1,1).contiguous())[None]
            print('Step:%d ,img1  NRMSE: %.3f, img2  NRMSE: %.3f'%(i, nrmse(gt_img_1, cplx_recon_1).item(),nrmse(gt_img_2, cplx_recon_2).item()))
   
    return x_next_1, x_next_2  


#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

