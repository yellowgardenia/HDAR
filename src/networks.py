"""
Networks for voxelmorph model

In general, these are fairly specific architectures that were designed for the presented papers.
However, the VoxelMorph concepts are not tied to a very particular architecture, and we 
encourage you to explore architectures that fit your needs. 
see e.g. more powerful unet function in https://github.com/adalca/neuron/blob/master/neuron/models.py
"""
# main imports
import sys

# third party
import numpy as np
import keras.backend as K
from keras.models import Model
import keras.layers as KL
from keras.layers import Conv3D, Activation, Input, UpSampling3D, concatenate
from keras.layers import LeakyReLU, Reshape, Lambda
from keras.initializers import RandomNormal
import keras.initializers
import tensorflow as tf
import tensorlayer as tl

# import neuron layers, which will be useful for Transforming.
sys.path.append('../ext/neuron')
sys.path.append('../ext/pynd-lib')
sys.path.append('../ext/pytools-lib')
import neuron.layers as nrn_layers
import neuron.utils as nrn_utils

# other vm functions
import losses


def attention_e_combine(e1, e3, nf, name='e-combine'):
    with tf.variable_scope(name):
        nf_e1 = e1.outputs.get_shape().as_list()[-1]
        nf_e3 = e3.outputs.get_shape().as_list()[-1]
        e1 = conv_block(e1, nf, strides=2, kernel=3, name='conv_e1')
        e3 = conv_block(e3, nf, strides=1, kernel=1, name='conv_e3')
        
        # attention
        e_combine = tl.layers.ConcatLayer([e1, e3], 4, name='e_combine')
        context_feature = conv_block(e_combine, nf, strides=1, kernel=1, name='context_feature')
        attn_map = conv_block(context_feature, nf, strides=1, kernel=3, name='conv_attn')
        attn_map = conv_block(attn_map, nf, strides=1, kernel=3, name='attention_map', act='sigmoid')
        attn_feature = tl.layers.ElementwiseLayer([context_feature, attn_map], combine_fn=tf.multiply, name='attention_feature')
        
        # refine
        ref_e1 = tl.layers.ConcatLayer([e1, attn_feature], 4, name='refine_e1_combine')
        a1 = conv_block(ref_e1, nf_e1, strides=1, kernel=3, name='ref_conv_e1')
        #a1 = tl.layers.DeConv3dLayer(a1, shape=[3,3,3,nf,nf_e1], strides=[1, 2, 2, 2, 1], padding='SAME', name='deconv_a1')
        a1.outputs = UpSampling3D()(a1.outputs)
        
        ref_e3 = tl.layers.ConcatLayer([e3, attn_feature], 4, name='refine_e3_combine')
        a3 = conv_block(ref_e3, nf, strides=1, kernel=3, name='ref_conv_e3')
        a3 = conv_block(a3, nf_e3, strides=1, kernel=1, name='conv_a3')
        return a1, a3, attn_map, attn_feature
        

def unet_core(x, is_train=False, name='UNet'):
    """
    unet architecture for voxelmorph models presented in the CVPR 2018 paper. 
    You may need to modify this code (e.g., number of layers) to suit your project needs.

    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the keras model
    """
    def keras_Up3dLayer(prev_layer, name='keras_upsampling_3d_layer'):
        with tf.variable_scope(name):
            net = tl.layers.LambdaLayer(prev_layer, fn=UpSampling3D(), name='upsampling_3d')
            return net

    enc_nf = [32, 64, 128, 256]  # [32,64]#[64,128]
    dec_nf = [256, 128, 128, 128, 64, 64]  # [64,64,64,32,32]

    with tf.variable_scope(name):
        x_in = tl.layers.InputLayer(x, name='inputs')
        
        conv1_1 = conv_block(x_in, enc_nf[0], name='conv1_1')
        conv1_2 = conv_block(conv1_1, enc_nf[1], 2, name='conv1_2')  # /2

        conv2_1 = conv_block(conv1_2, enc_nf[2], name='conv_2_1')
        conv2_2 = conv_block(conv2_1, enc_nf[3], 2, name='conv2_2')  # /4

        a1, a3, attn_map, attn_f = attention_e_combine(conv1_1, conv2_1, 16, name='e-combine')

        net = conv_block(conv2_2, dec_nf[0], name='conv_5')
        net = conv_block(net, dec_nf[1], name='conv_6')

        net = keras_Up3dLayer(net, name='up2')  # /2
        net = tl.layers.ConcatLayer([net, a3], 4, name='up2_concat')
        net = conv_block(net, dec_nf[2], name='up2_conv1')
        net = conv_block(net, dec_nf[3], name='up2_conv2')

        net = keras_Up3dLayer(net, name='up1')  # 1
        net = tl.layers.ConcatLayer([net, a1], 4, name='up1_concat')
        net = conv_block(net, dec_nf[4], name='up1_conv')

        net = conv_block(net, dec_nf[5], name='up1_conv2')
        
    return net, attn_map, attn_f


def unet_core_with_flow(x, flow, is_train=False, name='UNet-Flow'):
    """
    unet architecture for voxelmorph models presented in the CVPR 2018 paper. 
    You may need to modify this code (e.g., number of layers) to suit your project needs.

    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the keras model
    """
    def keras_Up3dLayer(prev_layer, name='keras_upsampling_3d_layer'):
        with tf.variable_scope(name):
            net = tl.layers.LambdaLayer(prev_layer, fn=UpSampling3D(), name='upsampling_3d')
            return net

    enc_nf = [32, 64, 128, 256]  # [32,64]#[64,128]
    dec_nf = [256, 128, 128, 128, 64, 64]  # [64,64,64,32,32]

    with tf.variable_scope(name):
        x_in = tl.layers.InputLayer(x, name='inputs')
        flow_in = tl.layers.InputLayer(flow, name='flow')
        
        conv1_1 = conv_block(x_in, enc_nf[0], name='conv1_1')
        conv1_2 = conv_block(conv1_1, enc_nf[1], 2, name='conv1_2')  # /2

        conv2_1 = conv_block(conv1_2, enc_nf[2], name='conv_2_1')
        conv2_2 = conv_block(conv2_1, enc_nf[3], 2, name='conv2_2')  # /4

        flow_1 = conv_block(flow_in, enc_nf[0], name='flow_1')  #1
        flow_2 = conv_block(flow_1, enc_nf[2], 2, name='flow_2') #1/2

        net = conv_block(conv2_2, dec_nf[0], name='conv_5')
        net = conv_block(net, dec_nf[1], name='conv_6')

        a1, a3, attn_map, attn_f = attention_e_combine(conv1_1, conv2_1, 16, name='e-combine')

        net = keras_Up3dLayer(net, name='up2')  # /2
        net = tl.layers.ConcatLayer([net, a3, flow_2], 4, name='up2_concat')
        net = conv_block(net, dec_nf[2], name='up2_conv1')
        net = conv_block(net, dec_nf[3], name='up2_conv2')

        net = keras_Up3dLayer(net, name='up1')  # 1
        net = tl.layers.ConcatLayer([net, a1, flow_1], 4, name='up1_concat')
        net = conv_block(net, dec_nf[4], name='up1_conv')

        net = conv_block(net, dec_nf[5], name='up1_conv2')
        
    return net, attn_map, attn_f


def unet_core_with_flow2(x, flow, is_train=False, name='UNet-Flow'):
    """
    unet architecture for voxelmorph models presented in the CVPR 2018 paper. 
    You may need to modify this code (e.g., number of layers) to suit your project needs.

    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the keras model
    """
    def keras_Up3dLayer(prev_layer, name='keras_upsampling_3d_layer'):
        with tf.variable_scope(name):
            net = tl.layers.LambdaLayer(prev_layer, fn=UpSampling3D(), name='upsampling_3d')
            return net

    enc_nf = [32, 64, 128, 256]  # [32,64]#[64,128]
    dec_nf = [256, 128, 128, 128, 64, 64]  # [64,64,64,32,32]

    with tf.variable_scope(name):
        x_in = tl.layers.InputLayer(x, name='inputs')
        flow_in = tl.layers.InputLayer(flow, name='flow')
        
        conv1_1 = conv_block(x_in, enc_nf[0], name='conv1_1')
        conv1_2 = conv_block(conv1_1, enc_nf[1], 2, name='conv1_2')  # /2

        conv2_1 = conv_block(conv1_2, enc_nf[2], name='conv_2_1')
        conv2_2 = conv_block(conv2_1, enc_nf[3], 2, name='conv2_2')  # /4

        flow_1 = conv_block(flow_in, enc_nf[0], name='flow_1')  #1
        flow_2 = conv_block(flow_1, enc_nf[2], 2, name='flow_2') #1/2

        net = conv_block(conv2_2, dec_nf[0], name='conv_5')
        net = conv_block(net, dec_nf[1], name='conv_6')

        a1, a3, attn_map, attn_f = attention_e_combine(conv1_1, conv2_1, 16, name='e-combine')

        net = keras_Up3dLayer(net, name='up2')  # /2
        net = tl.layers.ConcatLayer([net, conv2_1, flow_2], 4, name='up2_concat')
        net = conv_block(net, dec_nf[2], name='up2_conv1')
        net = conv_block(net, dec_nf[3], name='up2_conv2')

        net = keras_Up3dLayer(net, name='up1')  # 1
        net = tl.layers.ConcatLayer([net, conv1_1, flow_1], 4, name='up1_concat')
        net = conv_block(net, dec_nf[4], name='up1_conv')

        net = conv_block(net, dec_nf[5], name='up1_conv2')
        
    return net, attn_map, attn_f


def net_lr(src, tgt, mask_src, indexing='ij', is_train=False, reuse=None, name='net_lr'):
    """
    Subnetwork I--->Low Resolution--Large FoV.

    :param mask_src: seg or atlas of the corresponding img
    :return: the keras model
    """
    def keras_block(flow, src, interp_method='linear'):
        y = nrn_layers.SpatialTransformer(interp_method=interp_method, indexing=indexing)([src, flow])
        return y
        
    def transform_layer(flow, src, interp_method='linear', name='transform_layer'):
        with tf.variable_scope(name):
            y = tl.layers.LambdaLayer(
                flow, 
                fn=keras_block, 
                fn_args={
                    'src': src,
                    'interp_method': interp_method,
                    },
                name='keras_transform')
        return y

    with tf.variable_scope(name, reuse=reuse):
        x = tf.concat([src, tgt, src-tgt], 4, name='concat_input')
        unet_model, attn_map, attn_f = unet_core(x, is_train=is_train, name='UNet')
        

        flow = tl.layers.Conv3dLayer(
            unet_model, shape=(3,3,3,unet_model.outputs.get_shape().as_list()[-1],3), 
            strides=(1,1,1,1,1), padding='SAME', 
            W_init=tf.truncated_normal_initializer(stddev=1e-5), 
            b_init=tf.constant_initializer(value=0.0), name='flow')
        
        # warp the source with the flow
        y = transform_layer(flow, src, name='x_transform_layer')
        mask_y = transform_layer(flow, mask_src, interp_method='nearest', name='mask_transform_layer')

    return [y, mask_y, flow, attn_map.outputs, attn_f.outputs]


def net_mr(src, tgt, mask_src, flow, indexing='ij', is_train=False, reuse=None, name='net_mr'):
    """
    Middle FOV --> Middle Resolution

    :param vol_size: volume size. e.g. (224, 224, 224)
    :return: the keras model
    """
    def keras_block(flow, src, interp_method='linear'):
        y = nrn_layers.SpatialTransformer(interp_method=interp_method, indexing=indexing)([src, flow])
        return y
        
    def transform_layer(flow, src, interp_method='linear', name='transform_layer'):
        with tf.variable_scope(name):
            y = tl.layers.LambdaLayer(
                flow, 
                fn=keras_block, 
                fn_args={
                    'src': src,
                    'interp_method': interp_method,
                    },
                name='keras_transform')
        return y

    with tf.variable_scope(name, reuse=reuse):
        x = tf.concat([src, tgt, src-tgt], 4, name='concat_input')
        unet_model, attn_map, attn_f = unet_core_with_flow(x, flow, is_train=is_train, name='UNet')

        flow = tl.layers.Conv3dLayer(
            unet_model, shape=(3,3,3,unet_model.outputs.get_shape().as_list()[-1],3), 
            strides=(1,1,1,1,1), padding='SAME', 
            W_init=tf.truncated_normal_initializer(stddev=1e-5), 
            b_init=tf.constant_initializer(value=0.0), name='flow')
        
        # warp the source with the flow
        y = transform_layer(flow, src, name='x_transform_layer')
        mask_y = transform_layer(flow, mask_src, interp_method='nearest', name='mask_transform_layer')

    return [y, mask_y, flow, attn_map.outputs, attn_f.outputs]


def net_hr(src, tgt, mask_src, flow, indexing='ij', is_train=False, reuse=None, name='net_hr'):
    """
    Small FOV --> Large Resolution
    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the keras model
    """
    def keras_block(flow, src, interp_method='linear'):
        y = nrn_layers.SpatialTransformer(interp_method=interp_method, indexing=indexing)([src, flow])
        return y
        
    def transform_layer(flow, src, interp_method='linear', name='transform_layer'):
        with tf.variable_scope(name):
            y = tl.layers.LambdaLayer(
                flow, 
                fn=keras_block, 
                fn_args={
                    'src': src,
                    'interp_method': interp_method,
                    },
                name='keras_transform')
        return y

    with tf.variable_scope(name, reuse=reuse):
        x = tf.concat([src, tgt, src-tgt], 4, name='concat_input')
        unet_model, attn_map, attn_f = unet_core_with_flow2(x, flow, is_train=is_train, name='UNet')

        flow = tl.layers.Conv3dLayer(
            unet_model, shape=(3,3,3,unet_model.outputs.get_shape().as_list()[-1],3), 
            strides=(1,1,1,1,1), padding='SAME', 
            W_init=tf.truncated_normal_initializer(stddev=1e-5), 
            b_init=tf.constant_initializer(value=0.0), name='flow')

        gauss = make_gauss(k_size=3, sigma=1)#sigma=1
        zeros = np.zeros_like(gauss)
        gauss_kernal = tf.constant(np.array([[gauss,zeros,zeros],[zeros,gauss,zeros],[zeros,zeros,gauss]]),dtype=tf.float32)
        gauss_kernal = tf.transpose(gauss_kernal, [4,3,2,1,0])
        gauss_kernal= tf.cast(gauss_kernal, tf.float32)
        flow_smooth = tf.nn.conv3d(flow.outputs, gauss_kernal, strides=[1,1,1,1,1], padding='SAME')
        flow_s_tl = tl.layers.InputLayer(flow_smooth, name='flow_smooth_input')
        # warp the source with the flow
        y = transform_layer(flow_s_tl, src, name='x_transform_layer')
        mask_y = transform_layer(flow_s_tl, mask_src, interp_method='nearest', name='mask_transform_layer')

    return [y, mask_y, flow, attn_map.outputs, attn_f.outputs]


def make_gauss(k_size=3, sigma=1):
    center = k_size//2
    distance = np.zeros((k_size, k_size, k_size))
    for i in range(k_size):
        for j in range(k_size):
            for k in range(k_size):
                distance[i,j,k] = (center - i) * (center - i)+ (center - j) * (center - j) + (center - k) * (center - k)
    kernal = np.exp((0-distance)/(2*sigma*sigma))/2*np.pi*sigma*sigma
    kernal = kernal / kernal.sum()
    return kernal


def nn_trf(vol_size, indexing='xy'):
    """
    Simple transform model for nearest-neighbor based transformation
    Note: this is essentially a wrapper for the neuron.utils.transform(..., interp_method='nearest')
    """
    ndims = len(vol_size)

    # nn warp model
    subj_input = Input((*vol_size, 1), name='subj_input')
    trf_input = Input((*vol_size, ndims) , name='trf_input')

    # note the nearest neighbour interpolation method
    # note xy indexing because Guha's original code switched x and y dimensions
    nn_output = nrn_layers.SpatialTransformer(interp_method='nearest', indexing=indexing)
    nn_spatial_output = nn_output([subj_input, trf_input])
    return keras.models.Model([subj_input, trf_input], nn_spatial_output)


# Helper functions
def conv_block(x_in, nf, strides=1, kernel=3, name='Conv', act='lrelu'):
    """
    specific convolution module including convolution followed by leakyrelu
    """
    #ndims = len(x_in.get_shape()) - 2
    #assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    #Conv = getattr(KL, 'Conv%dD' % ndims)
    #x_out = Conv(nf, kernel_size=3, padding='same',
    #             kernel_initializer='he_normal', strides=strides)(x_in)
    #x_out = LeakyReLU(0.2)(x_out)
    w_init = tf.truncated_normal_initializer(stddev=1e-5)
    b_init = tf.constant_initializer(value=0.0)
    if act == 'lrelu':
        act_f = lambda x: tl.act.lrelu(x, 0.2)
    elif act == 'sigmoid':
        act_f = tf.nn.sigmoid

    c_in = x_in.outputs.get_shape().as_list()[-1]
    x_out = tl.layers.Conv3dLayer(x_in, shape=(kernel,kernel,kernel,c_in,nf), strides=(1,strides,strides,strides,1), padding='SAME', act=act_f, W_init=w_init, b_init=b_init, name=name)

    return x_out


def sample(args):
    """
    sample from a normal distribution
    """
    mu = args[0]
    log_sigma = args[1]
    noise = tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
    z = mu + tf.exp(log_sigma/2.0) * noise
    return z


def interp_upsampling(V):
    """ 
    upsample a field by a factor of 2
    TODO: should switch this to use neuron.utils.interpn()
    """

    grid = nrn_utils.volshape_to_ndgrid([f*2 for f in V.get_shape().as_list()[1:-1]])
    grid = [tf.cast(f, 'float32') for f in grid]
    grid = [tf.expand_dims(f/2 - f, 0) for f in grid]
    offset = tf.stack(grid, len(grid) + 1)

    # V = nrn_utils.transform(V, offset)
    V = nrn_layers.SpatialTransformer(interp_method='linear')([V, offset])
    return V

