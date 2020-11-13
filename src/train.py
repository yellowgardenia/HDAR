# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
train atlas-based alignment with CVPR2018 version of VoxelMorph 
"""

# python imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # set your device
import math
import glob
import sys
import random
from tqdm import trange
import SimpleITK as sitk
from argparse import ArgumentParser

# third-party imports
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import scipy
import scipy.ndimage as nd
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

# project imports
import datagenerators
import networks
import losses

sys.path.append('../ext/medipy-lib')
import medipy
from medipy.metrics import dice


def patch_selection_attn(beta, zoom_scales, limit_num=8, kernel=7, mi=20, ma=36, eps=1e-12):
    beta = np.mean(beta, axis=-1)
    det = []
    thresh = 0.5
    for x in range(mi, ma):
        for y in range(mi, ma):
            for z in range(mi, ma):
                # info = np.array([x, y, z, np.mean(att_map_array[x-8:x+8, y-8:y+8, z-8:z+8])])[np.newaxis, :]
                info = np.array([x-8, y-8, z-8, x+8, y+8, z+8, np.mean(beta[x-8:x+8, y-8:y+8, z-8:z+8])])[np.newaxis, :]
                det.append(info)

    det = np.concatenate(det, axis=0)

    x1 = det[:, 0]
    y1 = det[:, 1]
    z1 = det[:, 2]

    x2 = det[:, 3]
    y2 = det[:, 4]
    z2 = det[:, 5]

    scores = det[:, 6]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1) * (z2- z1 + 1)

    order = scores.argsort()[::-1]
    #
    keep = []

    goo = 0
    while order.size > 0:
        goo = goo + 1
        # print(goo)
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        zz1 = np.maximum(z1[i], z1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        zz2 = np.minimum(z2[i], z2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        d = np.maximum(0.0, zz2 - zz1 + 1)

        inter = w * h * d
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    a = det[keep, :]
    select_ps = []
    for i in range(len(keep)):
        x = (a[i,0]+a[i,3])//2
        y = (a[i,1]+a[i,4])//2
        z = (a[i,2]+a[i,5])//2
        if x <= mi or x >= ma:
            continue
        if y <= mi or y >= ma:
            continue
        if z <= mi or z >= ma:
            continue
        select_ps.append([x, y, x])

    sc_np = np.array(select_ps, np.float)
    optimize_selec_centers = []
    for k in range(len(select_ps)):
        optimize_selec_centers.append([int(np.round(sc_np[k,0]*zoom_scales[0])),
                                      int(np.round(sc_np[k,1]*zoom_scales[1])),
                                      int(np.round(sc_np[k,2]*zoom_scales[2]))])
    random.shuffle(optimize_selec_centers)
    print('Points number:', len(select_ps))
    return optimize_selec_centers[:limit_num]



def train(src_dir,
          tgt_dir,
          model_dir,
          model_lr_dir,
          lr,
          nb_epochs,
          reg_param,
          steps_per_epoch,
          batch_size,
          load_model_file=None,
          data_loss='ncc',
          initial_epoch=0):
    """
    model training function
    :param data_dir: folder with npz files for each subject.
    :param atlas_file: atlas filename. So far we support npz file with a 'vol' variable
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param model_dir: the model directory to save to
    :param lr: learning rate
    :param n_iterations: number of training iterations
    :param reg_param: the smoothness/reconstruction tradeoff parameter (lambda in CVPR paper)
    :param steps_per_epoch: frequency with which to save models
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    :param load_model_file: optional h5 model file to initialize with
    :param data_loss: data_loss: 'mse' or 'ncc
    """

    # prepare data files
    # for the CVPR and MICCAI papers, we have data arranged in train/validate/test folders
    # inside each folder is a /vols/ and a /asegs/ folder with the volumes
    # and segmentations. All of our papers use npz formated data.
    src_vol_names = glob.glob(os.path.join(src_dir, '*.npz'))
    tgt_vol_names = glob.glob(os.path.join(tgt_dir, '*.npz'))
    random.shuffle(src_vol_names)  # shuffle volume list
    random.shuffle(tgt_vol_names)  # shuffle volume list
    assert len(src_vol_names) > 0, "Could not find any training data"

    assert data_loss in ['mse', 'ncc'], 'Loss should be one of mse or cc, found %s' % data_loss
    if data_loss == 'ncc':
        data_loss = losses.NCC().loss

        # GPU handling
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    # set_session(tf.Session(config=config))

    vol_size = (56, 56, 56)
    # prepare the model
    src_lr = tf.placeholder(tf.float32, [None, *vol_size, 1], name='input_src_lr')
    tgt_lr = tf.placeholder(tf.float32, [None, *vol_size, 1], name='input_tgt_lr')
    srm_lr = tf.placeholder(tf.float32, [None, *vol_size, 1], name='mask_src_lr')
    attn_lr = tf.placeholder(tf.float32, [None, *vol_size, 1], name='attn_lr')

    src_mr = tf.placeholder(tf.float32, [None, *vol_size, 1], name='input_src_mr')
    tgt_mr = tf.placeholder(tf.float32, [None, *vol_size, 1], name='input_tgt_mr')
    srm_mr = tf.placeholder(tf.float32, [None, *vol_size, 1], name='mask_src_mr')
    df_lr2mr = tf.placeholder(tf.float32, [None, *vol_size, 3], name='df_lr2mr')
    attn_mr = tf.placeholder(tf.float32, [None, *vol_size, 1], name='attn_mr')

    src_hr = tf.placeholder(tf.float32, [None, *vol_size, 1], name='input_src_hr')
    tgt_hr = tf.placeholder(tf.float32, [None, *vol_size, 1], name='input_tgt_hr')
    srm_hr = tf.placeholder(tf.float32, [None, *vol_size, 1], name='mask_src_hr')
    df_mr2hr = tf.placeholder(tf.float32, [None, *vol_size, 3], name='df_mr2hr')
    attn_hr = tf.placeholder(tf.float32, [None, *vol_size, 1], name='attn_hr')

    model_lr = networks.net_lr(src_lr, tgt_lr, srm_lr)
    model_mr = networks.net_mr(src_mr, tgt_mr, srm_mr, df_lr2mr)
    model_hr = networks.net_hr(src_hr, tgt_hr, srm_hr, df_mr2hr)

    # the loss functions
    lr_ncc = data_loss(model_lr[0].outputs, tgt_lr)
    #lr_grd = losses.Grad('l2').loss(model_lr[0].outputs, model_lr[2].outputs)
    lr_grd = losses.Anti_Folding('l2').loss(model_lr[0].outputs, model_lr[2].outputs)

    cost_lr = lr_ncc + reg_param * lr_grd# + lr_attn

    mr_ncc = data_loss(model_mr[0].outputs, tgt_mr)
    #mr_grd = losses.Grad('l2').loss(model_mr[0].outputs, model_mr[2].outputs)
    mr_grd = losses.Anti_Folding('l2').loss(model_mr[0].outputs, model_mr[2].outputs)

    cost_mr = mr_ncc + reg_param * mr_grd

    hr_ncc = data_loss(model_hr[0].outputs, tgt_hr)
    #hr_grd = losses.Grad('l2').loss(model_hr[0].outputs, model_hr[2].outputs)
    hr_grd = losses.Anti_Folding('l2').loss(model_hr[0].outputs, model_hr[2].outputs)

    cost_hr = hr_ncc + reg_param * hr_grd

    # the training operations
    def get_v(name):
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if name in var.name]
        return d_vars
    #attn_vars = tl.layers.get_variables_with_name('cbam_1', True, True)
    attn_vars = get_v('cbam_1')
    for a_v in attn_vars:
        print(a_v)

    train_op_lr = tf.train.AdamOptimizer(lr).minimize(cost_lr)

    train_op_mr = tf.train.AdamOptimizer(lr).minimize(cost_mr)
    train_op_hr = tf.train.AdamOptimizer(lr).minimize(cost_hr)

    # data generator
    src_example_gen = datagenerators.example_gen(src_vol_names, batch_size=batch_size)
    tgt_example_gen = datagenerators.example_gen(tgt_vol_names, batch_size=batch_size)

    data_gen = datagenerators.gen_with_mask(src_example_gen, tgt_example_gen, batch_size=batch_size)

    variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['net_hr'])
    saver = tf.train.Saver(variables_to_restore)

    #saver = tf.train.Saver(max_to_keep=3)
    # fit generator
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # load initial weights
        try:
            if load_model_file is not None:
                model_file = tf.train.latest_checkpoint(load_model_file)  #
                saver.restore(sess, model_file)
        except:
            print('No files in', load_model_file)
        saver.save(sess, model_dir + 'dfnet', global_step=0)

        def resize_df(df, zoom):
            df1 = nd.interpolation.zoom(df[0, :, :, :, 0], zoom=zoom, mode='nearest', order=3) * zoom[
                0]  # Cubic: order=3; Bilinear: order=1; Nearest: order=0
            df2 = nd.interpolation.zoom(df[0, :, :, :, 1], zoom=zoom, mode='nearest', order=3) * zoom[1]
            df3 = nd.interpolation.zoom(df[0, :, :, :, 2], zoom=zoom, mode='nearest', order=3) * zoom[2]
            dfs = np.stack((df1, df2, df3), axis=3)
            return dfs[np.newaxis, :, :, :]

        class logPrinter(object):
            def __init__(self):
                self.n_batch = 0
                self.total_dice = []
                self.cost = []
                self.ncc = []
                self.grd = []

            def addLog(self, dice, cost, ncc, grd):
                self.n_batch += 1
                self.dice.append(dice)
                self.cost.append(cost)
                self.ncc.append(ncc)
                self.grd.append(grd)

            def output(self):
                dice = np.array(self.dice).mean(axis=0).round(3).tolist()
                cost = np.array(self.cost).mean()
                ncc = np.array(self.ncc).mean()
                grd = np.array(self.grd).mean()
                return dice, cost, ncc, grd, self.n_batch

            def clear(self):
                self.n_batch = 0
                self.dice = []
                self.cost = []
                self.ncc = []
                self.grd = []

        lr_log = logPrinter()
        mr_log = logPrinter()
        hr_log = logPrinter()

        # train low resolution
        # load initial weights
        saver = tf.train.Saver(max_to_keep=1)
        #if model_lr_dir is not None:
        #    model_lr_dir = tf.train.latest_checkpoint(model_lr_dir)  #
        #    print(model_lr_dir)
        #    saver.restore(sess, model_lr_dir)

        nb_epochs = 20#20#10
        steps_per_epoch = 30 * 29
        for epoch in range(nb_epochs):
            tbar = trange(steps_per_epoch, unit='batch', ncols=100)
            lr_log.clear()
            for i in tbar:
                image, mask = data_gen.__next__()
                global_X, global_atlas = image
                global_X_mask, global_atlas_mask = mask
                global_diff = global_X[0, :, :, :, 0]-global_atlas[0,:,:,:,0]

                # low resolution
                global_X_64 = nd.interpolation.zoom(global_X[0, :, :, :, 0], zoom=(0.25, 0.25, 0.25), mode='nearest')
                global_A_64 = nd.interpolation.zoom(global_atlas[0, :, :, :, 0], zoom=(0.25, 0.25, 0.25),
                                                    mode='nearest')
                global_XM_64 = nd.interpolation.zoom(global_X_mask[0, :, :, :, 0], zoom=(0.25, 0.25, 0.25),
                                                     mode='nearest', order=0)
                global_AM_64 = nd.interpolation.zoom(global_atlas_mask[0, :, :, :, 0], zoom=(0.25, 0.25, 0.25),
                                                     mode='nearest', order=0)
                global_diff_16 = nd.interpolation.zoom(global_diff, zoom=(0.25, 0.25, 0.25), mode='nearest')

                global_X_64 = global_X_64[np.newaxis, :, :, :, np.newaxis]
                global_A_64 = global_A_64[np.newaxis, :, :, :, np.newaxis]
                global_XM_64 = global_XM_64[np.newaxis, :, :, :, np.newaxis]
                global_AM_64 = global_AM_64[np.newaxis, :, :, :, np.newaxis]
                global_diff_16 = global_diff_16[np.newaxis, :, :, :, np.newaxis]


                feed_dict = {src_lr: global_X_64, tgt_lr: global_A_64, srm_lr: global_XM_64, attn_lr: global_diff_16}
                err_lr, _ = sess.run([cost_lr, train_op_lr], feed_dict=feed_dict)
                df_lr, warp_seg, elr_ncc, elr_grad, lr_attn_map, lr_attn_feature = sess.run(
                    [model_lr[2].outputs, model_lr[1].outputs, lr_ncc, lr_grd, model_lr[3], model_lr[4]], feed_dict=feed_dict)
                # print(df_lr.shape)
                lr_dice, _ = dice(warp_seg[0, :, :, :, 0], global_AM_64[0, :, :, :, 0], labels=[0, 10, 150, 250],
                                  nargout=2)
                lr_log.addLog(lr_dice, err_lr, elr_ncc, elr_grad)
                lr_out = lr_log.output()

                tbar.set_description('Epoch %d/%d ### step %i' % (epoch + 1, nb_epochs, i))
                tbar.set_postfix(lr_dice=lr_out[0], lr_cost=lr_out[1], lr_ncc=lr_out[2], lr_grd=lr_out[3])

            saver.save(sess, model_lr_dir + 'dfnet', global_step=0)
        # train middle resolution
        nb_epochs = 1#1
        steps_per_epoch = 30 * 29
        for epoch in range(nb_epochs):
            lr_log.clear()
            for lr_step in range(steps_per_epoch):
                image, mask = data_gen.__next__()
                global_X, global_atlas = image
                global_X_mask, global_atlas_mask = mask
                global_diff = global_X[0, :, :, :, 0]-global_atlas[0,:,:,:,0]

                # low resolution
                global_X_64 = nd.interpolation.zoom(global_X[0, :, :, :, 0], zoom=(0.25, 0.25, 0.25), mode='nearest')
                global_A_64 = nd.interpolation.zoom(global_atlas[0, :, :, :, 0], zoom=(0.25, 0.25, 0.25),
                                                    mode='nearest')
                global_XM_64 = nd.interpolation.zoom(global_X_mask[0, :, :, :, 0], zoom=(0.25, 0.25, 0.25),
                                                     mode='nearest', order=0)
                global_AM_64 = nd.interpolation.zoom(global_atlas_mask[0, :, :, :, 0], zoom=(0.25, 0.25, 0.25),
                                                     mode='nearest', order=0)
                global_diff_16 = nd.interpolation.zoom(global_diff, zoom=(0.25, 0.25, 0.25), mode='nearest')

                global_X_64 = global_X_64[np.newaxis, :, :, :, np.newaxis]
                global_A_64 = global_A_64[np.newaxis, :, :, :, np.newaxis]
                global_XM_64 = global_XM_64[np.newaxis, :, :, :, np.newaxis]
                global_AM_64 = global_AM_64[np.newaxis, :, :, :, np.newaxis]
                global_diff_16 = global_diff_16[np.newaxis, :, :, :, np.newaxis]

                feed_dict = {src_lr: global_X_64, tgt_lr: global_A_64, srm_lr: global_XM_64, attn_lr: global_diff_16}
                err_lr, _ = sess.run([cost_lr, train_op_lr], feed_dict=feed_dict)
                df_lr, warp_seg, elr_ncc, elr_grad, lr_attn_map, lr_attn_feature = sess.run(
                    [model_lr[2].outputs, model_lr[1].outputs, lr_ncc, lr_grd, model_lr[3], model_lr[4]], feed_dict=feed_dict)

                lr_dice, _ = dice(warp_seg[0, :, :, :, 0], global_AM_64[0, :, :, :, 0], labels=[0, 10, 150, 250],
                                  nargout=2)
                lr_log.addLog(lr_dice, err_lr, elr_ncc, elr_grad)
                lr_out = lr_log.output()

                print('\nEpoch %d/%d ### step %i' % (epoch + 1, nb_epochs, lr_out[-1]))
                print('[lr] lr_dice={}, lr_cost={:.3f}, lr_ncc={:.3f}, lr_grd={:.3f}'.format(lr_out[0], lr_out[1],
                                                                                             lr_out[2], lr_out[3]))

                # middle part
                df_lr_res2mr = resize_df(df_lr, zoom=(2, 2, 2))

                select_points_lr = patch_selection_attn(lr_attn_map, zoom_scales=[8, 8, 8], kernel=7, mi=10, ma=18)
                print(select_points_lr)
                mr_log.clear()

                for sp in select_points_lr:
                    mov_img_112 = global_X[0, sp[0] - 56:sp[0] + 56, sp[1] - 56:sp[1] + 56, sp[2] - 56:sp[2] + 56, 0]
                    fix_img_112 = global_atlas[0, sp[0] - 56:sp[0] + 56, sp[1] - 56:sp[1] + 56, sp[2] - 56:sp[2] + 56,
                                  0]
                    mov_seg_112 = global_X_mask[0, sp[0] - 56:sp[0] + 56, sp[1] - 56:sp[1] + 56, sp[2] - 56:sp[2] + 56,
                                  0]
                    fix_seg_112 = global_atlas_mask[0, sp[0] - 56:sp[0] + 56, sp[1] - 56:sp[1] + 56,
                                  sp[2] - 56:sp[2] + 56, 0]
                    dif_img_112 = global_diff[sp[0] - 56:sp[0] + 56, sp[1] - 56:sp[1] + 56, sp[2] - 56:sp[2] + 56]

                    #print(mov_img_112.shape)
                    if fix_img_112.shape != (112, 112, 112):
                        print(mov_img_112.shape)
                        continue
                    fix_112_56 = nd.interpolation.zoom(fix_img_112, zoom=(0.5, 0.5, 0.5), mode='nearest')
                    mov_112_56 = nd.interpolation.zoom(mov_img_112, zoom=(0.5, 0.5, 0.5), mode='nearest')
                    fix_112_56m = nd.interpolation.zoom(fix_seg_112, zoom=(0.5, 0.5, 0.5), mode='nearest', order=0)
                    mov_112_56m = nd.interpolation.zoom(mov_seg_112, zoom=(0.5, 0.5, 0.5), mode='nearest', order=0)
                    dif_112_56 = nd.interpolation.zoom(dif_img_112, zoom=(0.5, 0.5, 0.5), mode='nearest')

                    mid_fix_img = fix_112_56[np.newaxis, :, :, :, np.newaxis]
                    mid_mov_img = mov_112_56[np.newaxis, :, :, :, np.newaxis]
                    mid_fix_seg = fix_112_56m[np.newaxis, :, :, :, np.newaxis]
                    mid_mov_seg = mov_112_56m[np.newaxis, :, :, :, np.newaxis]
                    mid_dif_img = dif_112_56[np.newaxis, :, :, :, np.newaxis]
                    df_mr_feed = df_lr_res2mr[:,
                                 sp[0] // 2 - 28:sp[0] // 2 + 28,
                                 sp[1] // 2 - 28:sp[1] // 2 + 28,
                                 sp[2] // 2 - 28:sp[2] // 2 + 28,
                                 :]

                    feed_dict = {src_mr: mid_mov_img, tgt_mr: mid_fix_img, srm_mr: mid_mov_seg, df_lr2mr: df_mr_feed, attn_mr: mid_dif_img}
                    err_mr, _ = sess.run([cost_mr, train_op_mr], feed_dict=feed_dict)
                    df_mr, warp_seg, emr_ncc, emr_grad = sess.run(
                        [model_mr[2].outputs, model_mr[1].outputs, mr_ncc, mr_grd], feed_dict=feed_dict)

                    mr_dice, _ = dice(warp_seg[0, :, :, :, 0], mid_fix_seg[0, :, :, :, 0], labels=[0, 10, 150, 250],
                                      nargout=2)
                    mr_log.addLog(mr_dice, err_mr, emr_ncc, emr_grad)
                    mr_out = mr_log.output()

                    # print('  Epoch %d/%d ### step %i' % (epoch+1, nb_epochs, mr_out[-1]))
                    print('  [mr] {}/{} mr_dice={}, mr_cost={:.3f}, mr_ncc={:.3f}, mr_grd={:.3f}'.format(mr_out[-1],
                                                                                                         len(
                                                                                                             select_points_lr),
                                                                                                         mr_out[0],
                                                                                                         mr_out[1],
                                                                                                         mr_out[2],
                                                                                                         mr_out[3]))


            saver.save(sess, model_dir + 'dfnet', global_step=0)

        # train high resolution
        nb_epochs = 1
        steps_per_epoch = 300
        for epoch in range(nb_epochs):
            lr_log.clear()
            for lr_step in range(steps_per_epoch):
                image, mask = data_gen.__next__()
                global_X, global_atlas = image
                global_X_mask, global_atlas_mask = mask
                global_diff = global_X[0, :, :, :, 0]-global_atlas[0,:,:,:,0]

                # low resolution
                global_X_64 = nd.interpolation.zoom(global_X[0, :, :, :, 0], zoom=(0.25, 0.25, 0.25), mode='nearest')
                global_A_64 = nd.interpolation.zoom(global_atlas[0, :, :, :, 0], zoom=(0.25, 0.25, 0.25),
                                                    mode='nearest')
                global_XM_64 = nd.interpolation.zoom(global_X_mask[0, :, :, :, 0], zoom=(0.25, 0.25, 0.25),
                                                     mode='nearest', order=0)
                global_AM_64 = nd.interpolation.zoom(global_atlas_mask[0, :, :, :, 0], zoom=(0.25, 0.25, 0.25),
                                                     mode='nearest', order=0)
                global_diff_16 = nd.interpolation.zoom(global_diff, zoom=(0.25, 0.25, 0.25), mode='nearest')

                global_X_64 = global_X_64[np.newaxis, :, :, :, np.newaxis]
                global_A_64 = global_A_64[np.newaxis, :, :, :, np.newaxis]
                global_XM_64 = global_XM_64[np.newaxis, :, :, :, np.newaxis]
                global_AM_64 = global_AM_64[np.newaxis, :, :, :, np.newaxis]
                global_diff_16 = global_diff_16[np.newaxis, :, :, :, np.newaxis]

                feed_dict = {src_lr: global_X_64, tgt_lr: global_A_64, srm_lr: global_XM_64, attn_lr: global_diff_16}
                err_lr, _ = sess.run([cost_lr, train_op_lr], feed_dict=feed_dict)
                df_lr, warp_seg, elr_ncc, elr_grad, lr_attn_map, lr_attn_feature = sess.run(
                    [model_lr[2].outputs, model_lr[1].outputs, lr_ncc, lr_grd, model_lr[3], model_lr[4]], feed_dict=feed_dict)

                lr_dice, _ = dice(warp_seg[0, :, :, :, 0], global_AM_64[0, :, :, :, 0], labels=[0, 10, 150, 250],
                                  nargout=2)
                lr_log.addLog(lr_dice, err_lr, elr_ncc, elr_grad)
                lr_out = lr_log.output()

                print('\nEpoch %d/%d ### step %i' % (epoch + 1, nb_epochs, lr_out[-1]))
                print('[lr] lr_dice={}, lr_cost={:.3f}, lr_ncc={:.3f}, lr_grd={:.3f}'.format(lr_out[0], lr_out[1],
                                                                                             lr_out[2], lr_out[3]))

                # middle part
                df_lr_res2mr = resize_df(df_lr, zoom=(2, 2, 2))

                select_points_lr = patch_selection_attn(lr_attn_map, zoom_scales=[8, 8, 8], kernel=7, mi=10, ma=18)
                print(select_points_lr)
                mr_log.clear()

                for sp in select_points_lr:
                    mov_img_112 = global_X[0, sp[0] - 56:sp[0] + 56, sp[1] - 56:sp[1] + 56, sp[2] - 56:sp[2] + 56, 0]
                    fix_img_112 = global_atlas[0, sp[0] - 56:sp[0] + 56, sp[1] - 56:sp[1] + 56, sp[2] - 56:sp[2] + 56,
                                  0]
                    mov_seg_112 = global_X_mask[0, sp[0] - 56:sp[0] + 56, sp[1] - 56:sp[1] + 56, sp[2] - 56:sp[2] + 56,
                                  0]
                    fix_seg_112 = global_atlas_mask[0, sp[0] - 56:sp[0] + 56, sp[1] - 56:sp[1] + 56,
                                  sp[2] - 56:sp[2] + 56, 0]
                    dif_img_112 = global_diff[sp[0] - 56:sp[0] + 56, sp[1] - 56:sp[1] + 56, sp[2] - 56:sp[2] + 56]

                    #print(mov_img_112.shape)
                    if fix_img_112.shape != (112, 112, 112):
                        print(mov_img_112.shape)
                        continue
                    fix_112_56 = nd.interpolation.zoom(fix_img_112, zoom=(0.5, 0.5, 0.5), mode='nearest')
                    mov_112_56 = nd.interpolation.zoom(mov_img_112, zoom=(0.5, 0.5, 0.5), mode='nearest')
                    fix_112_56m = nd.interpolation.zoom(fix_seg_112, zoom=(0.5, 0.5, 0.5), mode='nearest', order=0)
                    mov_112_56m = nd.interpolation.zoom(mov_seg_112, zoom=(0.5, 0.5, 0.5), mode='nearest', order=0)
                    dif_112_56 = nd.interpolation.zoom(dif_img_112, zoom=(0.5, 0.5, 0.5), mode='nearest')

                    mid_fix_img = fix_112_56[np.newaxis, :, :, :, np.newaxis]
                    mid_mov_img = mov_112_56[np.newaxis, :, :, :, np.newaxis]
                    mid_fix_seg = fix_112_56m[np.newaxis, :, :, :, np.newaxis]
                    mid_mov_seg = mov_112_56m[np.newaxis, :, :, :, np.newaxis]
                    mid_dif_img = dif_112_56[np.newaxis, :, :, :, np.newaxis]
                    df_mr_feed = df_lr_res2mr[:,
                                 sp[0] // 2 - 28:sp[0] // 2 + 28,
                                 sp[1] // 2 - 28:sp[1] // 2 + 28,
                                 sp[2] // 2 - 28:sp[2] // 2 + 28,
                                 :]

                    feed_dict = {src_mr: mid_mov_img, tgt_mr: mid_fix_img, srm_mr: mid_mov_seg, df_lr2mr: df_mr_feed, attn_mr: mid_dif_img}
                    err_mr, _ = sess.run([cost_mr, train_op_mr], feed_dict=feed_dict)
                    df_mr, warp_seg, emr_ncc, emr_grad, mr_attn_map, mr_attn_feature = sess.run(
                        [model_mr[2].outputs, model_mr[1].outputs, mr_ncc, mr_grd, model_mr[3], model_mr[4]], feed_dict=feed_dict)

                    mr_dice, _ = dice(warp_seg[0, :, :, :, 0], mid_fix_seg[0, :, :, :, 0], labels=[0, 10, 150, 250],
                                      nargout=2)
                    mr_log.addLog(mr_dice, err_mr, emr_ncc, emr_grad)
                    mr_out = mr_log.output()

                    # print('  Epoch %d/%d ### step %i' % (epoch+1, nb_epochs, mr_out[-1]))
                    print('  [mr] {}/{} mr_dice={}, mr_cost={:.3f}, mr_ncc={:.3f}, mr_grd={:.3f}'.format(mr_out[-1],
                                                                                                         len(
                                                                                                             select_points_lr),
                                                                                                         mr_out[0],
                                                                                                         mr_out[1],
                                                                                                         mr_out[2],
                                                                                                         mr_out[3]))

                    # high part
                    df_mr_res2hr = resize_df(df_mr, zoom=(2, 2, 2))
                    hr_log.clear()
                    select_points_mr = patch_selection_attn(mr_attn_map, zoom_scales=[4, 4, 4], kernel=7, mi=8, ma=20)
                    print(30*'-')
                    print('High Part')
                    print(select_points_mr)
                    for spm in select_points_mr:
                        fix_img_56 = fix_img_112[spm[0] - 28:spm[0] + 28, spm[1] - 28:spm[1] + 28,
                                     spm[2] - 28:spm[2] + 28]
                        mov_img_56 = mov_img_112[spm[0] - 28:spm[0] + 28, spm[1] - 28:spm[1] + 28,
                                     spm[2] - 28:spm[2] + 28]
                        fix_seg_56 = fix_seg_112[spm[0] - 28:spm[0] + 28, spm[1] - 28:spm[1] + 28,
                                     spm[2] - 28:spm[2] + 28]
                        mov_seg_56 = mov_seg_112[spm[0] - 28:spm[0] + 28, spm[1] - 28:spm[1] + 28,
                                     spm[2] - 28:spm[2] + 28]
                        dif_img_56 = dif_img_112[spm[0] - 28:spm[0] + 28, spm[1] - 28:spm[1] + 28,
                                     spm[2] - 28:spm[2] + 28]
                        if fix_img_56.shape != (56, 56, 56):
                            continue

                        hig_fix_img = fix_img_56[np.newaxis, :, :, :, np.newaxis]
                        hig_mov_img = mov_img_56[np.newaxis, :, :, :, np.newaxis]
                        hig_fix_seg = fix_seg_56[np.newaxis, :, :, :, np.newaxis]
                        hig_mov_seg = mov_seg_56[np.newaxis, :, :, :, np.newaxis]
                        hig_dif_img = dif_img_56[np.newaxis, :, :, :, np.newaxis]

                        df_hr_feed = df_mr_res2hr[:, spm[0] - 28:spm[0] + 28, spm[1] - 28:spm[1] + 28,
                                     spm[2] - 28:spm[2] + 28, :]

                        feed_dict = {src_hr: hig_mov_img, tgt_hr: hig_fix_img, srm_hr: hig_mov_seg,
                                     df_mr2hr: df_hr_feed, attn_hr: hig_dif_img}
                        err_hr, _ = sess.run([cost_hr, train_op_hr], feed_dict=feed_dict)
                        df_hr, warp_seg, ehr_ncc, ehr_grad = sess.run(
                            [model_hr[2].outputs, model_hr[1].outputs, hr_ncc, hr_grd], feed_dict=feed_dict)

                        hr_dice, _ = dice(warp_seg[0, :, :, :, 0], hig_fix_seg[0, :, :, :, 0], labels=[0, 10, 150, 250],
                                          nargout=2)
                        hr_log.addLog(hr_dice, err_hr, ehr_ncc, ehr_grad)
                        hr_out = hr_log.output()

                        # print('  Epoch %d/%d ### step %i' % (epoch+1, nb_epochs, mr_out[-1]))
                        print(
                            '    [hr] {}/{} hr_dice={}, hr_cost={:.3f}, hr_ncc={:.3f}, hr_grd={:.3f}'.format(hr_out[-1],
                                                                                                             len(
                                                                                                                 select_points_mr),
                                                                                                             hr_out[0],
                                                                                                             hr_out[1],
                                                                                                             hr_out[2],
                                                                                                             hr_out[3]))

                saver.save(sess, model_dir + 'dfnet', global_step=lr_step)


if __name__ == "__main__":
    train(
        src_dir='path/to/image',
        tgt_dir='path/to/image',
        model_dir='../my_models/',
        model_lr_dir ='../lr_models/',
        lr=1e-4,
        nb_epochs=1000,
        reg_param=1.5,
        steps_per_epoch=30,
        batch_size=1,
        load_model_file='../my_models/', #'../my_models/',#'../max_w/',
        data_loss='ncc')
