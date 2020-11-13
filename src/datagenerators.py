"""
data generators for VoxelMorph

for the CVPR and MICCAI papers, we have data arranged in train/validate/test folders
inside each folder is a /vols/ and a /asegs/ folder with the volumes
and segmentations. All of our papers use npz formated data.
"""

import os
import numpy as np
import random

	
def load_example_by_name(vol_name, seg_name):

    X = np.load(vol_name)['vol']
    X = np.reshape(X, (1,) + X.shape + (1,))

    return_vals = [X]

    X_seg = np.load(vol_name)['seg']
    X_seg = np.reshape(X_seg, (1,) + X_seg.shape + (1,))
    return_vals.append(X_seg)

    return tuple(return_vals)


def example_gen(vol_names, batch_size=1):
    #idx = 0
    while True:
        idxes = np.random.randint(len(vol_names), size=batch_size)

        X_data = []
        X_data_mask = []
        for idx in idxes:
            X = np.load(vol_names[idx])['vol']
            X_seg = np.load(vol_names[idx])['seg']
            X = np.reshape(X, (1,) + X.shape + (1,))
            X_seg = np.reshape(X_seg, (1,) + X_seg.shape + (1,))
            X_data += [X]
            X_data_mask += [X_seg]

        if batch_size > 1:
            return_vals = [(np.concatenate(X_data, 0), np.concatenate(X_data_mask, 0))]
        else:
            return_vals = [(X_data[0], X_data_mask[0])]

        yield tuple(return_vals)


def gen_with_mask(gen_src, gen_tgt, batch_size=1):
    """ generator used for cvpr 2018 """
    while True:
        X1, X_mask1= next(gen_src)[0]
        X2, X_mask2= next(gen_tgt)[0]
        X1, X2, X_mask1, X_mask2 = augment(X1, X2, X_mask1, X_mask2)
        yield ([X1, X2], [X_mask1, X_mask2])


def augment(mov, fix, mov_seg, fix_seg):
    axis = [1,2,3]
    random.shuffle(axis)
    mov = np.transpose(mov, axes=(0, axis[0], axis[1], axis[2], 4))
    fix = np.transpose(fix, axes=(0, axis[0], axis[1], axis[2], 4))
    mov_seg = np.transpose(mov_seg, axes=(0, axis[0], axis[1], axis[2], 4))
    fix_seg = np.transpose(fix_seg, axes=(0, axis[0], axis[1], axis[2], 4))

    if np.random.randint(2) == 1:
        mov = mov[:,::-1]
        fix = fix[:,::-1]
        mov_seg = mov_seg[:,::-1]
        fix_seg = fix_seg[:,::-1]

    if np.random.randint(2) == 1:
        mov = mov[:,:,::-1]
        fix = fix[:,:,::-1]
        mov_seg = mov_seg[:,:,::-1]
        fix_seg = fix_seg[:,:,::-1]

    if np.random.randint(2) == 1:
        mov = mov[:,:,:,::-1]
        fix = fix[:,:,:,::-1]
        mov_seg = mov_seg[:,:,:,::-1]
        fix_seg = fix_seg[:,:,:,::-1]

    return mov, fix, mov_seg, fix_seg


def load_example_with_roi(vol_name):
    X = np.load(vol_name)['vol']
    X = np.reshape(X, (1,) + X.shape + (1,))

    return_vals = [X]

    X_atlas = np.load(vol_name)['atlas']
    X_atlas = np.reshape(X_atlas, (1,) + X_atlas.shape + (1,))
    return_vals.append(X_atlas)

    return tuple(return_vals)
