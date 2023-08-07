import numpy as np
from torch.utils.data import Dataset
import glob
import os
import torch
import sigpy as sp
import h5py
import random


def adjoint_fs(ksp,maps):
    coil_imgs = sp.ifft(ksp, axes = (-2,-1))
    coil_imgs_with_maps = coil_imgs*np.conj(maps)
    img_out = np.sum(coil_imgs_with_maps, axis = -3)
    return img_out

def forward_fs(img,maps):
    coil_imgs = img*maps
    coil_ksp = sp.fft(coil_imgs, axes = (-2,-1))
    return coil_ksp

class SKM_TEA_mri(Dataset):
    def __init__(self, target_files, center_slice=256, num_slices=100, R=4):
        self.target_files  = target_files
        self.num_slices    = num_slices
        self.center_slice  = center_slice
        self.ACS           = 12
        self.noise_lvl     = 0.01
        self.crop_sz       = 160
        self.R             = R

    def __len__(self):
        return len(self.target_files) * self.num_slices

    def __getitem__(self, idx):

        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Separate slice and sample
        sample_idx = idx // self.num_slices
        slice_idx  = self.center_slice + np.mod(idx, self.num_slices) - self.num_slices // 2

        # Load MRI samples and maps
        with h5py.File(self.target_files[sample_idx], 'r') as contents:
            # Get k-space for specific slice
            target = np.asarray(contents['target'][slice_idx,:,:,:,0])# shape = [H,W,2]

        img1 = target[:,:,0]
        img2 = target[:,:,1]
        img1 = sp.ifft(sp.resize(sp.fft(img1,axes=(-2,-1)),(self.crop_sz,self.crop_sz)), axes=(-2,-1))
        img2 = sp.ifft(sp.resize(sp.fft(img2,axes=(-2,-1)),(self.crop_sz,self.crop_sz)), axes=(-2,-1))
        
        sigma = self.noise_lvl*np.percentile(abs(img1),99)

        

        norm_const = np.percentile(np.abs(target), 99)
        noise_1 = sigma*(np.random.randn(160,160)+1j*np.random.randn(160,160))
        noise_2 = sigma*(np.random.randn(160,160)+1j*np.random.randn(160,160))


        outer_line_count = int(int(self.crop_sz/self.R) - self.ACS)
        mask1 = np.zeros((self.crop_sz,self.crop_sz))
        mask2 = np.zeros((self.crop_sz,self.crop_sz))
        # use 14 lines for center of ksp
        center_idx = np.arange(int(self.crop_sz/2)-int(self.ACS/2),int(self.crop_sz/2)+int(self.ACS/2-1)) # 16 central lines for ACS
        total_idx = np.arange(self.crop_sz)
        rem_lines = np.delete(total_idx, center_idx)

        random.shuffle(rem_lines)
        mask_lines_1=np.concatenate((center_idx,rem_lines[0:outer_line_count]))
        random.shuffle(rem_lines)
        mask_lines_2=np.concatenate((center_idx,rem_lines[0:outer_line_count]))

        # print(mask_lines)
        mask1[:,mask_lines_1] = 1
        mask2[:,mask_lines_2] = 1

        mask1_batched = mask1[None,None]
        img1_batched = img1[None, None]
        mask2_batched = mask2[None,None]
        img2_batched = img2[None, None]

        maps1_batched = np.ones((1,1,1,1))
        ksp1_batched = sp.fft(img1_batched+noise_1, axes=(-2,-1))*mask1

        maps2_batched = np.ones((1,1,1,1))
        ksp2_batched = sp.fft(img2_batched+noise_2, axes=(-2,-1))*mask2

        norm = np.percentile(abs(sp.ifft(ksp1_batched, axes = (-2,-1))), 99)


        sample = {'mask_1': mask1_batched[0],
        'maps_1': maps1_batched[0],
        'gt_img_1': img1_batched[0]/norm,
        'ksp_1':ksp1_batched[0]/norm,

        'mask_2': mask2_batched[0],
        'maps_2': maps2_batched[0],
        'gt_img_2': img2_batched[0]/norm,
        'ksp_2':ksp2_batched[0]/norm,
        
        'norm':norm}

        return sample



class fastMRI_knee(Dataset):
    def __init__(self, target_files, center_slice=17, num_slices=20, ACS_perc=0.03, R=4):
        self.target_files  = target_files
        self.num_slices    = num_slices
        self.center_slice  = center_slice
        self.ACS_per       = ACS_perc
        self.noise_lvl     = 0.01
        self.crop_sz       = 320
        self.R             = R

    def __len__(self):
        return len(self.target_files) * self.num_slices

    def __getitem__(self, idx):
        # print('entered get_item')
        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Separate slice and sample
        sample_idx = idx // self.num_slices
        slice_idx  = self.center_slice + np.mod(idx, self.num_slices) - self.num_slices // 2

        with h5py.File(self.target_files[sample_idx], 'r') as cont:
            PD_img  = np.asarray(cont['PD_imgs'])[slice_idx]
            PDFS_img  = np.asarray(cont['PDFS_imgs'])[slice_idx]

        # print('loaded h5 successvully')
        PD_img = sp.resize(PD_img,(self.crop_sz, self.crop_sz))
        PDFS_img = sp.resize(PDFS_img,(self.crop_sz,self.crop_sz))

        sigma = .01*np.percentile(abs(PD_img),99)

        noise_1 = sigma*(np.random.randn(self.crop_sz,self.crop_sz)+1j*np.random.randn(self.crop_sz,self.crop_sz))
        noise_2 = sigma*(np.random.randn(self.crop_sz,self.crop_sz)+1j*np.random.randn(self.crop_sz,self.crop_sz))

        ACS = int(self.ACS_per*self.crop_sz)
        outer_line_count = int(self.crop_sz/self.R) - ACS
        mask1 = np.zeros((self.crop_sz,self.crop_sz))
        mask2 = np.zeros((self.crop_sz,self.crop_sz))
        
        # use 14 lines for center of ksp
        center_idx = np.arange(int(self.crop_sz/2)-int(ACS/2),int(self.crop_sz/2)+int(ACS/2-1)) # 16 central lines for ACS
        total_idx = np.arange(self.crop_sz)
        rem_lines = np.delete(total_idx, center_idx)
        
        random.shuffle(rem_lines)
        mask_lines_1=np.concatenate((center_idx,rem_lines[0:outer_line_count]))
        random.shuffle(rem_lines)
        mask_lines_2=np.concatenate((center_idx,rem_lines[0:outer_line_count]))

        scale = np.percentile(abs(PD_img), 99)

        mask1[:,mask_lines_1] = 1
        mask2[:,mask_lines_2] = 1

        mask1_batched = mask1[None]
        img1_batched = PD_img[None]
        mask2_batched = mask2[None]
        img2_batched = PDFS_img[None]

        # print('greated_masks')
        maps1_batched = np.ones((1,1,1))
        ksp1_batched = sp.fft(img1_batched, axes=(-2,-1))*mask1

        maps2_batched = np.ones((1,1,1))
        ksp2_batched = sp.fft(img2_batched, axes=(-2,-1))*mask2

        norm1 = np.max(abs(sp.ifft(ksp1_batched, axes = (-2,-1))))
        norm2 = np.max(abs(sp.ifft(ksp2_batched, axes = (-2,-1))))

        sample = {'mask_1': mask1_batched,
        'maps_1': maps1_batched,
        'gt_img_1': img1_batched/norm1,
        'ksp_1':ksp1_batched/norm1,

        'mask_2': mask2_batched,
        'maps_2': maps2_batched,
        'gt_img_2': img2_batched/norm2,
        'ksp_2':ksp2_batched/norm2,
        
        'norm1':norm1,
        'norm2':norm2}

        return sample
