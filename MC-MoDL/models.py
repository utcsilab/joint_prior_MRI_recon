#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import sigpy as sp
import numpy as np
import copy as copy

from core_ops import TorchHybridSense, TorchHybridImage
from core_ops import TorchMoDLSense, TorchMoDLImage
from utils import fft, ifft

from opt import ZConjGrad
from unet import NormUnet, Unet
from resnet import ResNet

class MoDL(torch.nn.Module):
    def __init__(self, hparams):
        super(MoDL, self).__init__()
        # Storage
        self.verbose    = hparams.verbose
        self.batch_size = hparams.batch_size
        self.block2_max_iter = hparams.max_cg_steps
        self.cg_eps          = hparams.cg_eps

        # Logging
        self.logging  = hparams.logging

        # ImageNet parameters
        self.img_channels = hparams.img_channels
        self.img_blocks   = hparams.img_blocks
        self.img_arch     = hparams.img_arch

        # Attention parameters
        self.att_config   = hparams.att_config
        if hparams.img_arch != 'Unet':
            self.latent_channels = hparams.latent_channels
            self.kernel_size     = hparams.kernel_size

        # Get useful values
        self.ones_mask = torch.ones((1)).cuda()

        # Initialize trainable parameters
        if hparams.l2lam_train:
            self.block2_l2lam = torch.nn.Parameter(torch.tensor(
                    hparams.l2lam_init *
                    np.ones((1))).cuda())
        else:
            self.block2_l2lam = torch.tensor(
                    hparams.l2lam_init *
                    np.ones((1))).cuda()

        self.image_net = Unet(in_chans=2, out_chans=2, chans=self.img_channels,num_pool_layers=self.img_blocks)

    # Get torch operators for the entire batch
    def get_core_torch_ops(self, mps_kernel, img_kernel, mask, direction):
        # List of output ops
        normal_ops, adjoint_ops, forward_ops = [], [], []

        # For each sample in batch
        for idx in range(self.batch_size):
            # Type
            if direction == 'ConvSense':
                forward_op, adjoint_op, normal_op = \
                    TorchMoDLSense(mps_kernel[idx], mask[idx])
            elif direction == 'ConvImage':
                forward_op, adjoint_op, normal_op = \
                    TorchMoDLImage(img_kernel[idx], mask[idx])

            # Add to lists
            normal_ops.append(normal_op)
            adjoint_ops.append(adjoint_op)
            forward_ops.append(forward_op)

        # Return operators
        return normal_ops, adjoint_ops, forward_ops

    # Given a batch of inputs and ops, get a single batch operator
    def get_batch_op(self, input_ops, batch_size):
        # Inner function trick
        def core_function(x):
            # Store in list
            output_list = []
            for idx in range(batch_size):
                output_list.append(input_ops[idx](x[idx])[None, ...])
            # Stack and return
            return torch.cat(output_list, dim=0)
        return core_function

    def forward(self, data, contrast, meta_unrolls=1):
        if contrast == 1:
            mask1      = data['mask_1']
            ksp1       = data['ksp_1']
            # Initializers
            with torch.no_grad():
                maps1 = data['maps_1']
        elif contrast ==2:
            mask1      = data['mask_2']
            ksp1       = data['ksp_2']
            # Initializers
            with torch.no_grad():
                maps1 = data['maps_2']

        if self.logging:
            img1_logs = []

        normal_ops_1, adjoint_ops_1, forward_ops_1 = \
            self.get_core_torch_ops(maps1, None,
                    mask1, 'ConvSense')

        # Get joint batch operators for adjoint and normal
        normal_batch_op_1, adjoint_batch_op_1 = \
            self.get_batch_op(normal_ops_1, self.batch_size), \
            self.get_batch_op(adjoint_ops_1, self.batch_size)


        # get initial image x = A^H(y)
        est_img_kernel_1 = adjoint_batch_op_1(ksp1) #flipped order of was ksp[:,mask_idx]: same below
        # print(est_img_kernel.shape)
        # For each outer unroll
        for meta_idx in range(meta_unrolls):
            # Convert to reals
            # est_img_kernel_prev = est_img_kernel.clone()
            est_img_kernel_1 = torch.view_as_real(est_img_kernel_1)# shape: [B,H,W,2]

            est_img_kernel =  est_img_kernel_1
            # Apply image denoising network in image space
            # stack images to be 4 channel input
            est_img_kernel = self.image_net(est_img_kernel.permute(0,-1,-3,-2)) + est_img_kernel.permute(0,-1,-3,-2)
            # Convert to complex
            est_img_kernel = est_img_kernel.permute(0,-2,-1,1).contiguous() # shape: [B,H,W,4]
            # # Convert to complex
            # est_img_kernel = torch.view_as_complex(est_img_kernel)
            est_img_kernel_1 = torch.view_as_complex(est_img_kernel)

            rhs_1 = adjoint_batch_op_1(ksp1) + \
                self.block2_l2lam[0] * est_img_kernel_1

            # Get unrolled CG op
            cg_op_1 = ZConjGrad(rhs_1, normal_batch_op_1,
                             l2lam=self.block2_l2lam[0],
                             max_iter=self.block2_max_iter,
                             eps=self.cg_eps, verbose=self.verbose)
            # Run CG
            est_img_kernel_1 = cg_op_1(est_img_kernel_1)

            # Log
            if self.logging:
                img1_logs.append(est_img_kernel_1)


        if self.logging:
            return est_img_kernel_1, img1_logs
        else:
            return est_img_kernel_1

class joint_MoDL(torch.nn.Module):
    def __init__(self, hparams):
        super(joint_MoDL, self).__init__()
        # Storage
        self.verbose    = hparams.verbose
        self.batch_size = hparams.batch_size
        self.block2_max_iter = hparams.max_cg_steps
        self.cg_eps          = hparams.cg_eps

        # Logging
        self.logging  = hparams.logging

        # ImageNet parameters
        self.img_channels = hparams.img_channels
        self.img_blocks   = hparams.img_blocks
        self.img_arch     = hparams.img_arch

        # Attention parameters
        self.att_config   = hparams.att_config
        if hparams.img_arch != 'Unet':
            self.latent_channels = hparams.latent_channels
            self.kernel_size     = hparams.kernel_size

        # Get useful values
        self.ones_mask = torch.ones((1)).cuda()

        # Initialize trainable parameters
        if hparams.l2lam_train:
            self.block2_l2lam = torch.nn.Parameter(torch.tensor(
                    hparams.l2lam_init *
                    np.ones((1))).cuda())
        else:
            self.block2_l2lam = torch.tensor(
                    hparams.l2lam_init *
                    np.ones((1))).cuda()

        self.image_net = Unet(in_chans=4, out_chans=4, chans=self.img_channels,num_pool_layers=self.img_blocks)

    # Get torch operators for the entire batch
    def get_core_torch_ops(self, mps_kernel, img_kernel, mask, direction):
        # List of output ops
        normal_ops, adjoint_ops, forward_ops = [], [], []

        # For each sample in batch
        for idx in range(self.batch_size):
            # Type
            if direction == 'ConvSense':
                forward_op, adjoint_op, normal_op = \
                    TorchMoDLSense(mps_kernel[idx], mask[idx])
            elif direction == 'ConvImage':
                forward_op, adjoint_op, normal_op = \
                    TorchMoDLImage(img_kernel[idx], mask[idx])

            # Add to lists
            normal_ops.append(normal_op)
            adjoint_ops.append(adjoint_op)
            forward_ops.append(forward_op)

        # Return operators
        return normal_ops, adjoint_ops, forward_ops

    # Given a batch of inputs and ops, get a single batch operator
    def get_batch_op(self, input_ops, batch_size):
        # Inner function trick
        def core_function(x):
            # Store in list
            output_list = []
            for idx in range(batch_size):
                output_list.append(input_ops[idx](x[idx])[None, ...])
            # Stack and return
            return torch.cat(output_list, dim=0)
        return core_function

    def forward(self, data, meta_unrolls=1):
        mask1      = data['mask_1']
        ksp1       = data['ksp_1']
        mask2      = data['mask_2']
        ksp2       = data['ksp_2']
        # print(mask1.shape)
        # print(ksp1.shape)
        # Initializers
        with torch.no_grad():
            maps1 = data['maps_1']
            maps2 = data['maps_2']

        if self.logging:
            img1_logs = []
            img2_logs = []

        normal_ops_1, adjoint_ops_1, forward_ops_1 = \
            self.get_core_torch_ops(maps1, None,
                    mask1, 'ConvSense')
        normal_ops_2, adjoint_ops_2, forward_ops_2 = \
            self.get_core_torch_ops(maps2, None,
                    mask2, 'ConvSense')
        # Get joint batch operators for adjoint and normal
        normal_batch_op_1, adjoint_batch_op_1 = \
            self.get_batch_op(normal_ops_1, self.batch_size), \
            self.get_batch_op(adjoint_ops_1, self.batch_size)
        normal_batch_op_2, adjoint_batch_op_2 = \
            self.get_batch_op(normal_ops_2, self.batch_size), \
            self.get_batch_op(adjoint_ops_2, self.batch_size)

        # get initial image x = A^H(y)
        est_img_kernel_1 = adjoint_batch_op_1(ksp1) #flipped order of was ksp[:,mask_idx]: same below
        est_img_kernel_2 = adjoint_batch_op_2(ksp2)
        # print(est_img_kernel.shape)
        # For each outer unroll
        for meta_idx in range(meta_unrolls):
            # Convert to reals
            # est_img_kernel_prev = est_img_kernel.clone()
            est_img_kernel_1 = torch.view_as_real(est_img_kernel_1[:,0])# shape: [B,H,W,2]
            est_img_kernel_2 = torch.view_as_real(est_img_kernel_2[:,0])# shape: [B,H,W,2]
            # print(est_img_kernel_1.shape)
            est_img_kernel =  torch.cat((est_img_kernel_1, est_img_kernel_2), dim=-1) # shape: [B,H,W,4]
            # print(est_img_kernel.shape)
            # Apply image denoising network in image space
            # stack images to be 4 channel input
            est_img_kernel = self.image_net(est_img_kernel.permute(0,-1,-3,-2).float()) + est_img_kernel.permute(0,-1,-3,-2).float()
            # Convert to complex
            est_img_kernel = est_img_kernel.permute(0,-2,-1,1).contiguous() # shape: [B,H,W,4]
            # # Convert to complex
            # est_img_kernel = torch.view_as_complex(est_img_kernel)
            est_img_kernel_1 = torch.view_as_complex(est_img_kernel[...,0:2])
            est_img_kernel_2 = torch.view_as_complex(est_img_kernel[...,2:])

            rhs_1 = adjoint_batch_op_1(ksp1) + \
                self.block2_l2lam[0] * est_img_kernel_1
            rhs_2 = adjoint_batch_op_2(ksp2) + \
                self.block2_l2lam[0] * est_img_kernel_2

            # Get unrolled CG op
            cg_op_1 = ZConjGrad(rhs_1, normal_batch_op_1,
                             l2lam=self.block2_l2lam[0],
                             max_iter=self.block2_max_iter,
                             eps=self.cg_eps, verbose=self.verbose)
            cg_op_2 = ZConjGrad(rhs_2, normal_batch_op_2,
                             l2lam=self.block2_l2lam[0],
                             max_iter=self.block2_max_iter,
                             eps=self.cg_eps, verbose=self.verbose)
            # Run CG
            est_img_kernel_1 = cg_op_1(est_img_kernel_1)
            est_img_kernel_2 = cg_op_1(est_img_kernel_2)

            # Log
            if self.logging:
                img1_logs.append(est_img_kernel_1)
                img2_logs.append(est_img_kernel_2)


        if self.logging:
            return est_img_kernel_1, est_img_kernel_2, img1_logs, img2_logs
        else:
            return est_img_kernel_1, est_img_kernel_2
