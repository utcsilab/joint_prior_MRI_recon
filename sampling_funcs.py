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


#----------------------------------------------------------------------------
# Generalized ablation sampler, representing the superset of all sampling
# methods discussed in the paper.
def marginal_ablation_sampler(
    y_meas, mask, s_maps, l_ss, net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, num_inner_steps = 4, sigma_min=None, sigma_max=None, rho=7,
    solver='euler', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, gt_img=None, verbose = True,
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        for inner_step in range(num_inner_steps):
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
            t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
            x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

            # Euler step on Prior.
            h = t_next - t_hat
            denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
            d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
            x_prime = x_hat + alpha * h * d_cur
            t_prime = t_hat + alpha * h

            # Euler step on liklihood
            # x_cur.shape = [1,2,H,W]
            x_hat_cplx = torch.view_as_complex(x_hat.permute(0,-2,-1,1).contiguous())[None] #shape: [1,1,H,W]
            Ax = forward(image=x_hat_cplx, maps=s_maps, mask=mask)
            res = Ax-y_meas
            l_cur = adjoint(ksp=res, maps=s_maps, mask=mask)[:,0,...]
            
            # print(l_cur.shape)


            #Apply 2nd order correction.
            if solver == 'euler' or i == num_steps - 1:
                x_next = x_hat + h * d_cur - l_ss*torch.view_as_real(l_cur).permute(0,-1,1,2)
            else:
                assert solver == 'heun'
                denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels).to(torch.float64)
                d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
                x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)


            if verbose:
                cplx_recon = torch.view_as_complex(x_next.permute(0,-2,-1,1).contiguous())[None] #shape: [1,1,H,W]
                print('Step:%d ,  NRMSE: %.3f'%(i, nrmse(gt_img, cplx_recon).item()))
    return x_next

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
        denoised = net(x_hat, t_cur, class_labels).to(torch.float64)
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


def posterior_sampler_edm(net, gt_img, y, maps, mask, latents, l_ss=1.0, second_order=False,
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

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        x_hat = x_hat.requires_grad_() #starting grad tracking with the noised img

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

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

        # Apply optional 2nd order correction.
        if second_order and i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        if verbose:
            cplx_recon = torch.view_as_complex(x_next.permute(0,-2,-1,1).contiguous())[None]
            print('Step:%d , NRMSE: %.3f'%(i, nrmse(gt_img, cplx_recon).item()))
            if i%10==0:
                img_stack.append(cplx_recon.cpu().numpy())

    return x_next, img_stack


    return x_next

def joint_posterior_sampler(net, gt_img_1, y_1, maps_1, mask_1, gt_img_2, y_2, maps_2, mask_2, latents, l_ss=1.0, second_order=False,
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

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        # x_hat = x_hat #starting grad tracking with the noised img
        x_hat_1 = x_hat[:,0:2,...]
        x_hat_2 = x_hat[:,2:,...]
        x_hat_1 = x_hat_1.requires_grad_()
        x_hat_2 = x_hat_2.requires_grad_()
        x_hat_new = torch.cat((x_hat_1, x_hat_2), dim=1)
        # print(x_hat_new.shape)
        # Euler step.
        denoised = net(x_hat_new, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat_new - denoised) / t_hat
        x_next = x_hat_new + (t_next - t_hat) * d_cur

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

        # Apply optional 2nd order correction.
        if second_order and i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        if verbose:
            cplx_recon_1 = torch.view_as_complex(x_next_1.permute(0,-2,-1,1).contiguous())[None]
            cplx_recon_2 = torch.view_as_complex(x_next_2.permute(0,-2,-1,1).contiguous())[None]
            print('Step:%d ,img1  NRMSE: %.3f, img2  NRMSE: %.3f'%(i, nrmse(gt_img_1, cplx_recon_1).item(),nrmse(gt_img_2, cplx_recon_2).item()))
   
    return x_next_1, x_next_2    

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


def conditional_posterior_sampler(net, contrast, gt_img_1, y_1, maps_1, mask_1, gt_img_2, y_2, maps_2, mask_2, latents, l_ss=1.0, second_order=False,
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

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)

        if contrast=='PDFS':
            x_cur[:,:2,...] = gt_img_1
        elif contrast=='PD':
            x_cur[:,:2,...] = gt_img_2

        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        # x_hat = x_hat #starting grad tracking with the noised img
        x_hat_1 = x_hat[:,0:2,...]
        x_hat_2 = x_hat[:,2:,...]

        if contrast=='PD':
            x_hat_1 = x_hat_1.requires_grad_()
        if contrast=='PDFS':
            x_hat_2 = x_hat_2.requires_grad_()
        x_hat_new = torch.cat((x_hat_1, x_hat_2), dim=1)
        # print(x_hat_new.shape)
        # Euler step.
        denoised = net(x_hat_new, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat_new - denoised) / t_hat
        x_next = x_hat_new + (t_next - t_hat) * d_cur

        # Likelihood step
        denoised_cplx_1 = torch.view_as_complex(denoised[:,0:2,...].permute(0,-2,-1,1).contiguous())[None]
        denoised_cplx_2 = torch.view_as_complex(denoised[:,2:,...].permute(0,-2,-1,1).contiguous())[None]

        Ax_1 = forward(image=denoised_cplx_1, maps=maps_1, mask=mask_1)
        Ax_2 = forward(image=denoised_cplx_2, maps=maps_2, mask=mask_2)
        residual_1 = y_1 - Ax_1
        residual_2 = y_2 - Ax_2
        sse_1 = torch.norm(residual_1)**2
        sse_2 = torch.norm(residual_2)**2


        if contrast=='PD':
            likelihood_score_1 = torch.autograd.grad(outputs=sse_1, inputs=x_hat_1, retain_graph=True)[0]
            x_next[:,0:2,...] = x_next[:,0:2,...] - (l_ss / torch.sqrt(sse_1)) * likelihood_score_1
            x_hat_1 = x_hat_1.requires_grad_(False)
        if contrast=='PDFS':
            likelihood_score_2 = torch.autograd.grad(outputs=sse_2, inputs=x_hat_2, retain_graph=True)[0]
            x_next[:,2:,...] = x_next[:,2:,...] - (l_ss / torch.sqrt(sse_2)) * likelihood_score_2
            x_hat_2 = x_hat_2.requires_grad_(False)

        x_next_1 = x_next[:,0:2,...]
        x_next_2 = x_next[:,2:,...]
        # Cleanup 
        x_next = x_next.detach()

        # Apply optional 2nd order correction.
        if second_order and i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

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



def posterior_sampler_look_ahead(net, gt_img, y, maps, mask, latents, l_ss=1.0, second_order=False,
                     class_labels=None, randn_like=torch.randn_like, num_steps=100, look_ahead_steps = 1,
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

        # look ahead 
        # (1) probably should recalculate the time points again but thats for a later date)
        # (2) maybe we could use a second order approximator for the look ahead
        

        t_la_steps = t_steps[i:i+look_ahead_steps+1]
        x_next_la = x_next
        x_cur_la = x_hat

        for j, (t_la_cur, t_la_next) in enumerate(zip(t_la_steps[:-1], t_la_steps[1:])):
            denoised_la = net(x_cur_la, t_la_cur, class_labels).to(torch.float64)
            d_la_cur = (x_cur_la - denoised_la)/t_la_cur
            # take step over prior score and add noise
            # x_next_la = x_hat + (t_la_next - t_la_cur) * d_la_cur #+ ((2*t_cur)*(t_cur-t_next))**0.5 * randn_like(x_cur)
            x_next_la = x_cur_la + (t_la_next - t_la_cur) * d_la_cur
            x_cur_la = x_next_la



        denoised_la_cplx = torch.view_as_complex(denoised_la.permute(0,-2,-1,1).contiguous())[None]
        Ax = forward(image=denoised_la_cplx, maps=maps, mask=mask)
        residual = y - Ax
        sse = torch.norm(residual)**2
        likelihood_score = torch.autograd.grad(outputs=sse, inputs=x_hat)[0]
        likelihood_step =  (l_ss / torch.sqrt(sse)) * likelihood_score
        # Cleanup 
        x_hat = x_hat.requires_grad_(False)




        # Euler step on prior score.
        denoised = net(x_hat, t_cur, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised)/t_cur
        # take step over prior score and add noise
        x_next = x_hat + (t_next - t_cur) * d_cur #+ ((2*t_cur)*(t_cur-t_next))**0.5 * randn_like(x_cur)

        # Euler step on 
        x_next = x_next - likelihood_step


        # Cleanup 
        x_next = x_next.detach()
        x_hat = x_hat.requires_grad_(False)


        if verbose:
            cplx_recon = torch.view_as_complex(x_next.permute(0,-2,-1,1).contiguous())[None]
            print('Step:%d , NRMSE: %.3f'%(i, nrmse(gt_img, cplx_recon).item()))
            if i%10==0:
                img_stack.append(cplx_recon.cpu().numpy())

    return x_next, img_stack



def posterior_sampler_look_ahead_mod(net, gt_img, y, maps, mask, latents, l_ss=1.0, second_order=False,
                     class_labels=None, randn_like=torch.randn_like, num_steps=100, look_ahead_steps = 1,
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

        # look ahead 
        # (1) probably should recalculate the time points again but thats for a later date)
        # (2) maybe we could use a second order approximator for the look ahead
        
        # calculate step sizes to be used in look ahead loop (basically just change the sigma_max to be the current time step in the outer loop)
        la_step_indices = torch.arange(look_ahead_steps, dtype=torch.float64, device=latents.device)
        # t_la_steps = (t_cur ** (1 / rho) + la_step_indices / (look_ahead_steps - 1) * (sigma_min ** (1 / rho) - t_cur ** (1 / rho))) ** rho
        t_la_steps = (t_cur ** (1 / rho) + la_step_indices / (look_ahead_steps) * (sigma_min ** (1 / rho) - t_cur ** (1 / rho))) ** rho
        t_la_steps = torch.cat([net.round_sigma(t_la_steps), torch.zeros_like(t_la_steps[:1])]) # t_N = 0
        
        x_next_la = x_next
        x_cur_la = x_hat

        print(t_la_steps)
        for j, (t_la_cur, t_la_next) in enumerate(zip(t_la_steps[:-1], t_la_steps[1:])):
            denoised_la = net(x_cur_la, t_la_cur, class_labels).to(torch.float64)
            d_la_cur = (x_cur_la - denoised_la)/t_la_cur
            # take step over prior score and add noise
            x_next_la = x_cur_la + (t_la_next - t_la_cur) * d_la_cur #+ ((2*t_cur)*(t_cur-t_next))**0.5 * randn_like(x_cur)
            x_cur_la = x_next_la



        denoised_la_cplx = torch.view_as_complex(denoised_la.permute(0,-2,-1,1).contiguous())[None]
        Ax = forward(image=denoised_la_cplx, maps=maps, mask=mask)
        residual = y - Ax
        sse = torch.norm(residual)**2
        likelihood_score = torch.autograd.grad(outputs=sse, inputs=x_hat)[0]
        likelihood_step =  (l_ss / torch.sqrt(sse)) * likelihood_score
        # Cleanup 
        x_hat = x_hat.requires_grad_(False)




        # Euler step on prior score.
        denoised = net(x_hat, t_cur, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised)/t_cur
        # take step over prior score and add noise
        x_next = x_hat + (t_next - t_cur) * d_cur #+ ((2*t_cur)*(t_cur-t_next))**0.5 * randn_like(x_cur)

        # Euler step on 
        x_next = x_next - likelihood_step


        # Cleanup 
        x_next = x_next.detach()
        x_hat = x_hat.requires_grad_(False)


        if verbose:
            cplx_recon = torch.view_as_complex(x_next.permute(0,-2,-1,1).contiguous())[None]
            print('Step:%d , NRMSE: %.3f'%(i, nrmse(gt_img, cplx_recon).item()))
            if i%10==0:
                img_stack.append(cplx_recon.cpu().numpy())

    return x_next, img_stack


def look_ahead_loop(net, x_next, x_cur, t_s, class_labels, num_steps,ord='1st'):
    for j, (t_cur, t_next) in enumerate(zip(t_s[:-1], t_s[1:])):
        denoised = net(x_cur, t_cur, class_labels).to(torch.float64)
        d_cur = (x_cur - denoised)/t_cur
        # take step over prior score and add noise
        x_next = x_cur + (t_next - t_cur) * d_cur #+ ((2*t_cur)*(t_cur-t_next))**0.5 * randn_like(x_cur)
        x_cur = x_next

        # if ord=='2nd' and j < num_steps - 1:
        #     denoised = net(x_next, t_next, class_labels).to(torch.float64)
        #     d_prime = (x_next - denoised) / t_next
        #     x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_cur
