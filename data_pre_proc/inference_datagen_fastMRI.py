import h5py, os
import numpy as np
import glob
import torch
from tqdm import tqdm
import sigpy as sp
import matplotlib.pyplot as plt
import random

R = 12

crop_sz = 320

folder = glob.glob('/csiNAS/brett/fastmri_contrast_pairs_singlecoil/*.h5')[-5:]



for samp in range(len(folder)):
    target_file = folder[samp]
    for slice in range(20):
        with h5py.File(target_file, 'r') as cont:
            # print(cont.keys())
            # ksp = np.asarray(cont['kspace'])[:,:,40]
            # maps = np.asarray(cont['maps'])
            # masks = np.asarray(cont['masks'])
            PD_imgs  = np.asarray(cont['PD_imgs'])
            PDFS_imgs  = np.asarray(cont['PDFS_imgs'])

        PD_img = sp.resize(PD_imgs[slice+9],(crop_sz,crop_sz))
        PDFS_img = sp.resize(PDFS_imgs[slice+9],(crop_sz,crop_sz))

        sigma = .01*np.percentile(abs(PD_imgs),99)

        noise_1 = sigma*(np.random.randn(crop_sz,crop_sz)+1j*np.random.randn(crop_sz,crop_sz))
        noise_2 = sigma*(np.random.randn(crop_sz,crop_sz)+1j*np.random.randn(crop_sz,crop_sz))

        ACS = int(.08*320)
        outer_line_count = int(int(crop_sz/R) - ACS)
        mask1 = np.zeros((crop_sz,crop_sz))
        mask2 = np.zeros((crop_sz,crop_sz))
        # use 14 lines for center of ksp
        center_idx = np.arange(int(crop_sz/2)-int(ACS/2),int(crop_sz/2)+int(ACS/2-1)) # 16 central lines for ACS
        total_idx = np.arange(crop_sz)
        rem_lines = np.delete(total_idx, center_idx)

        random.shuffle(rem_lines)
        mask_lines_1=np.concatenate((center_idx,rem_lines[0:outer_line_count]))
        random.shuffle(rem_lines)
        mask_lines_2=np.concatenate((center_idx,rem_lines[0:outer_line_count]))

        # print(mask_lines)
        mask1[:,mask_lines_1] = 1
        mask2[:,mask_lines_2] = 1

        mask1_batched = mask1[None,None]
        img1_batched = PD_img[None, None]
        mask2_batched = mask2[None,None]
        img2_batched = PDFS_img[None, None]

        maps1_batched = np.ones((1,1,1,1))
        ksp1_batched = sp.fft(img1_batched, axes=(-2,-1))*mask1

        maps2_batched = np.ones((1,1,1,1))
        ksp2_batched = sp.fft(img2_batched, axes=(-2,-1))*mask2

        norm1 = np.max(abs(sp.ifft(ksp1_batched, axes = (-2,-1))))
        norm2 = np.max(abs(sp.ifft(ksp2_batched, axes = (-2,-1))))

        dict = {'mask_1': mask1_batched,
        'maps_1': maps1_batched,
        'gt_img_1': img1_batched,
        'ksp_1':ksp1_batched,

        'mask_2': mask2_batched,
        'maps_2': maps2_batched,
        'gt_img_2': img2_batched,
        'ksp_2':ksp2_batched,
        
        'norm1':norm1,
        'norm2':norm2
}

        file = '/home/blevac/cond_score_data/fastMRI_knee/sample%d_R%d.pt'%(samp*20 + slice,R)

        torch.save(dict,file)



