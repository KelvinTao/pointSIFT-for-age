import os
import sys

import tensorflow as tf
import tf_utils.tf_util as tf_util
from tf_utils.pointSIFT_util import pointSIFT_module, pointSIFT_res_module, pointnet_fp_module, pointnet_sa_module


def placeholder_inputs(batch_size,num_point,num_class):
    #num_class=numofclasses*2
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_class//2,2))
    smpws_pl = tf.placeholder(tf.float32, shape=(batch_size))
    return pointclouds_pl, labels_pl, smpws_pl


def get_model(point_cloud, is_training, num_class, bn_decay=None, feature=None):
    """ Semantic segmentation PointNet, input is B x N x 3, output B x num_class """
    end_points = {}
    l0_xyz = point_cloud
    l0_points = feature
    end_points['l0_xyz'] = l0_xyz

    # c0: 1024*128
    c0_l0_xyz, c0_l0_points, c0_l0_indices = pointSIFT_res_module(l0_xyz, l0_points, radius=0.1, out_channel=64, is_training=is_training, bn_decay=bn_decay, scope='layer0_c0', merge='concat')
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(c0_l0_xyz, c0_l0_points, npoint=1024, radius=0.1, nsample=32, mlp=[64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')

    # c1: 256*256
    c0_l1_xyz, c0_l1_points, c0_l1_indices = pointSIFT_res_module(l1_xyz, l1_points, radius=0.25, out_channel=128, is_training=is_training, bn_decay=bn_decay, scope='layer1_c0')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(c0_l1_xyz, c0_l1_points, npoint=256, radius=0.2, nsample=32, mlp=[128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')

    # c2: 256*512
    c0_l2_xyz, c0_l2_points, c0_l2_indices = pointSIFT_res_module(l2_xyz, l2_points, radius=0.5, out_channel=256, is_training=is_training, bn_decay=bn_decay, scope='layer2_c0')
    c1_l2_xyz, c1_l2_points, c1_l2_indices = pointSIFT_res_module(c0_l2_xyz, c0_l2_points, radius=0.5, out_channel=512, is_training=is_training, bn_decay=bn_decay, scope='layer2_c1', same_dim=True)
    l2_cat_points = tf.concat([c0_l2_points, c1_l2_points], axis=-1)
    fc_l2_points = tf_util.conv1d(l2_cat_points, 512, 1, padding='VALID', bn=True, is_training=is_training, scope='layer2_conv_c2', bn_decay=bn_decay)

    # c3: 64*512
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(c1_l2_xyz, fc_l2_points, npoint=64, radius=0.4, nsample=32, mlp=[512,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')

    
    # FC layers:64*256->64*128---8192
    net = tf_util.conv1d(l3_points, 256, 1, padding='VALID', bn=True, is_training=is_training, scope='layer4_conv', bn_decay=bn_decay)
    #net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='layer5_conv', bn_decay=bn_decay)
    ##flatten:B*8192
    flat = tf.reshape(net, [-1,64*128])
    ##dense layer:4096
    dense = tf_util.fully_connected(flat,4096,scope='layer6_dense',bn=True,bn_decay=bn_decay,is_training=is_training)
    dense = tf_util.fully_connected(dense,4096,scope='layer7_dense',bn=True,bn_decay=bn_decay,is_training=is_training)
    dense = tf_util.dropout(dense, keep_prob=0.5, is_training=is_training, scope='dp')
    logits = tf_util.fully_connected(dense,num_class,scope='layer8_dense',bn=True,bn_decay=bn_decay,is_training=is_training)#logits
    return logits, end_points

    '''
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [512,512], is_training, bn_decay, scope='fa_layer2')
    _, l2_points_1, _ = pointSIFT_module(l2_xyz, l2_points, radius=0.5, out_channel=512, is_training=is_training, bn_decay=bn_decay, scope='fa_layer2_c0')
    _, l2_points_2, _ = pointSIFT_module(l2_xyz, l2_points, radius=0.5, out_channel=512, is_training=is_training, bn_decay=bn_decay, scope='fa_layer2_c1')
    _, l2_points_3, _ = pointSIFT_module(l2_xyz, l2_points, radius=0.5, out_channel=512, is_training=is_training, bn_decay=bn_decay, scope='fa_layer2_c2')

    l2_points = tf.concat([l2_points_1, l2_points_2, l2_points_3], axis=-1)
    l2_points = tf_util.conv1d(l2_points, 512, 1, padding='VALID', bn=True, is_training=is_training, scope='fa_2_fc', bn_decay=bn_decay)

    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,256], is_training, bn_decay, scope='fa_layer3')
    _, l1_points_1, _ = pointSIFT_module(l1_xyz, l1_points, radius=0.25, out_channel=256, is_training=is_training, bn_decay=bn_decay, scope='fa_layer3_c0')
    _, l1_points_2, _ = pointSIFT_module(l1_xyz, l1_points_1, radius=0.25, out_channel=256, is_training=is_training, bn_decay=bn_decay, scope='fa_layer3_c1')
    l1_points = tf.concat([l1_points_1, l1_points_2], axis=-1)
    l1_points = tf_util.conv1d(l1_points, 256, 1, padding='VALID', bn=True, is_training=is_training, scope='fa_1_fc', bn_decay=bn_decay)

    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,128], is_training, bn_decay, scope='fa_layer4')
    _, l0_points, _ = pointSIFT_module(l0_xyz, l0_points, radius=0.1, out_channel=128, is_training=is_training, bn_decay=bn_decay, scope='fa_layer4_c0')


    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, num_class, 1, padding='VALID', activation_fn=None, scope='fc2')

    return net, end_points
    '''


def get_loss(logits,labels,num_class,smpws=1,reg=1):
    """
    :param logits: Bx(C*2)--Bx200(100*2)
    :param labels: BxCx2--Bx100x2
    :param smpw: B ; sample weight
    :num_class:200 --class_number*2
    """
    part_logits=tf.reshape(logits,[-1,num_class//2,2])
    classify_loss=tf.reduce_mean(tf.multiply(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=part_logits, labels=labels),1),smpws))
    regularization_loss=tf.reduce_mean(tf.losses.get_regularization_losses())
    classify_loss+=reg*regularization_loss  ##scalar
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss

def eval_pred(logits,input_labels,num_class,wt=1):#预测结果评估
    """
    :param logits: Bx(C*2)--Bx200(100*2);>< age_thresh
    """
    part_logits=tf.reshape(logits,[-1,num_class//2,2])
    part_logits1=tf.map_fn(lambda x:x[:,0],tf.nn.softmax(part_logits))
    pred=tf.reduce_sum(part_logits1,1)
    labb=tf.reduce_sum(tf.map_fn(lambda x:x[:,0],input_labels),1)
    mae_wt=tf.reduce_mean(tf.multiply(tf.abs(pred-labb),wt))
    mae=tf.reduce_mean(tf.abs(pred-labb))
    return pred,mae,mae_wt


'''
def get_loss(pred, label, smpw):
    """
    :param pred: BxNxC
    :param label: BxN
    :param smpw: BxN
    :return:
    """
    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred, weights=smpw)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss
'''