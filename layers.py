import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

def global_activation(inputs):
    return tf.nn.relu(inputs)

def init_conv(inputs, filters, filter_size=7, train = True):
    inputs = tf.layers.conv2d(inputs, filters, filter_size, 2, 'same', name='init_conv_1')
    inputs = tf.layers.batch_normalization(inputs, training=train)
    inputs = global_activation(inputs)
    inputs = tf.layers.max_pooling2d(inputs, 3, 2, 'same')
    return inputs

def res_identity_block(inputs, small_filters, last_filters, kernel_size = 3, strides=2,
                       train=True, name="res_identity_block"):
    name = "res_identity_block"
    # short_cut
    inputs_short_cut = tf.layers.conv2d(inputs, last_filters, 1, strides, 'same', name=name + "/conv_short_cut")
    inputs_short_cut = tf.layers.batch_normalization(inputs_short_cut, training=train)
    # block
    inputs_block = tf.layers.conv2d(inputs, small_filters, 1, strides, 'same', name=name+"/conv_1")
    inputs_block = tf.layers.batch_normalization(inputs_block, training=train)
    inputs_block = global_activation(inputs_block)
    inputs_block = tf.layers.conv2d(inputs_block, small_filters, kernel_size    , 1, 'same', name=name+"/conv_2")
    inputs_block = tf.layers.batch_normalization(inputs_block, training=train)
    inputs_block = global_activation(inputs_block)
    inputs_block = tf.layers.conv2d(inputs_block, last_filters, 1, 1, 'same', name=name+"/conv_3")
    inputs_block = tf.layers.batch_normalization(inputs_block, training=train)
    result = global_activation(tf.add(inputs_block, inputs_short_cut))
    return result

def res_conv_block(inputs, small_filters, last_filters, kernel_size=3, train=True, index=0, name="res_conv_block"):
    name = name + "res_conv_block" + str(index)
    inputs_short_cut = inputs
    # block
    inputs_block = tf.layers.conv2d(inputs, small_filters, 1, 1, 'same', name=name+"/conv_1")
    inputs_block = tf.layers.batch_normalization(inputs_block, training=train)
    inputs_block = global_activation(inputs_block)
    inputs_block = tf.layers.conv2d(inputs_block, small_filters, kernel_size, 1, 'same', name=name+"/conv_2")
    inputs_block = tf.layers.batch_normalization(inputs_block, training=train)
    inputs_block = global_activation(inputs_block)
    inputs_block = tf.layers.conv2d(inputs_block, last_filters, 1, 1, 'same', name=name+"/conv_3")
    inputs_block = tf.layers.batch_normalization(inputs_block, training=train)
    result = global_activation(tf.add(inputs_block, inputs_short_cut))
    return result

def res_block(inputs, layer_num, small_filters, last_filters, kernel_size=3, train=True, name="redisual_layer"):
    conv2d = inputs
    if last_filters != inputs.get_shape()[-1]:
        conv2d = tf.layers.conv2d(inputs, last_filters, kernel_size, 1, 'same')
    for i in range(layer_num):
            conv2d = res_conv_block(conv2d, small_filters, last_filters, kernel_size,
                              train, index=i, name=name)
    return conv2d

def res_transition_down(inputs, small_filters, last_filters, kernel_size=3, strides=2,
                        is_dropout=True, dropout=0.2, train=True):
    inputs = res_identity_block(inputs, small_filters, last_filters, kernel_size, strides, train)
    if is_dropout:
        inputs = tf.layers.dropout(inputs, dropout)
    return inputs

def base_conv(inputs, filters, filter_size=3, dropout=0.2,dilation_rate=1, train = True, name="base_dense"):
    inputs = tf.layers.batch_normalization(inputs, name=name+'/batch_norm')
    inputs = tf.nn.relu(inputs)
    inputs = tf.layers.conv2d(inputs, filters, filter_size, 1, 'same', dilation_rate=dilation_rate, name=name+"/conv2d")
    return inputs


# def dense_block(inputs, layer_num, grow_filters=12, filter_size=3, dropout=0.2, train=True, up=False, dilation_rate=1, name = "dense_block"):
#     stake = inputs
#     filters = inputs.get_shape()[-1]
#     for i in range(layer_num):
#         with tf.name_scope("dense_block"+str(i)):
#             if up:
#                 filters -= grow_filters
#             else:
#                 filters += grow_filters
#             stake_new = base_dense(stake, grow_filters, filter_size, dropout, dilation_rate=dilation_rate,
#                                    train=train, name=name+"/baseDense."+str(i))
#             stake = tf.concat([stake, stake_new], -1)
#     return stake

def dense_block(inputs, layer_num, grow_filters=12, filter_size=3, dropout=0.2, train=True, up=False, dilation_rate=1, name = "dense_block"):
    stake = inputs
    for i in range(layer_num):
        with tf.name_scope("dense_block"+str(i)):
            stake_new = base_conv(stake, grow_filters * 4, 1, dropout, dilation_rate=dilation_rate,
                                   train=train, name=name + "/baseDense1x1." + str(i))
            stake_new = base_conv(stake_new, grow_filters, 3, dropout, dilation_rate=dilation_rate,
                                   train=train, name=name+"/baseDense3x3"+str(i))
            stake = tf.concat([stake, stake_new], -1)
    return stake

def redisual(inputs, filters, kernel_size = 3, dropout=0.2,
             train=True, dilation_rate=1, name="redisual"):
    inputs = tf.layers.batch_normalization(inputs, name=name+"/batch_norm", training=train)
    shot_cut = inputs

    conv2d = tf.layers.conv2d(inputs, 64, 1, 1,
                              'same',activation=tf.nn.relu, dilation_rate=dilation_rate, name=name+"/conv_1")

    conv2d = tf.layers.conv2d(conv2d, 64, kernel_size, 1,
                              'same',activation=tf.nn.relu, dilation_rate=dilation_rate, name=name+"/conv_2")

    conv2d = tf.layers.conv2d(conv2d, filters, 1, 1,
                              'same', activation=tf.nn.relu, dilation_rate=dilation_rate, name=name + "/conv_3")

    # conv2d = tf.layers.conv2d(inputs, 64, kernel_size, 1,
    #                           'same',activation=tf.nn.relu, dilation_rate=dilation_rate, name=name+"/conv_1")
    #
    # conv2d = tf.layers.conv2d(conv2d, 64, kernel_size, 1,
    #                           'same',activation=tf.nn.relu, dilation_rate=dilation_rate, name=name+"/conv_2")
    #
    # conv2d = tf.layers.conv2d(conv2d, filters, kernel_size, 1,
    #                           'same', activation=tf.nn.relu, dilation_rate=dilation_rate, name=name + "/conv_3")

    return tf.add(conv2d, shot_cut)


def redisual_block(inputs, layer_num, filters , kernel_size=3, dropout=0.2, train=True,
                   dilation_rate=1, name="redisual_layer"):
    conv2d = inputs

    if filters != inputs.get_shape()[-1]:
        conv2d = tf.layers.conv2d(inputs, filters, kernel_size,
                                    1, 'same', dilation_rate=dilation_rate)
        #conv2d = max_attention_conv(conv2d, filters, 5, 1, 'same',name=name+"_pre")
    for i in range(layer_num):
        #inputs_max_pool = tf.layers.max_pooling2d(conv2d, 5, 1, 'same')

        conv2d = redisual(conv2d, filters, kernel_size,
                          dropout, train, dilation_rate=dilation_rate, name = name+'/redisual_block'+str(i))

        #conv2d = tf.add(conv2d, inputs_max_pool)
        #conv2d = tf.concat([conv2d, inputs_max_pool],axis=-1)
        #conv2d = tf.layers.conv2d(conv2d, filters, 1, 1, 'same', name=name + '/max_attention' + str(i))
        #conv2d = max_attention_conv(conv2d, filters,5, 1, 'same', name= name+'/redisual_block'+str(i))
    return conv2d

def max_attention_conv(inputs, filters, pool_size, strides, padding='valid',name='/max_attention'):
    inputs_max_pool = tf.layers.max_pooling2d(inputs, pool_size, strides, 'same')
    inputs = tf.concat([inputs, inputs_max_pool],axis=-1)
    inputs = tf.layers.conv2d(inputs, filters, 1, 1, 'same',name =name + '/max_attention')
    return inputs

def fuzzy_attention(fuzzy_features, normal_features, filters, pool_size, strides, name='fuzzy_attention_layer'):
    inputs_0 = tf.layers.conv2d(fuzzy_features, filters, 1, 1, 'same', name=name + '/fuzzy_attention_conv')
    # method1029_15
    inputs = tf.layers.max_pooling2d(inputs_0, pool_size, strides, 'same')
    # method1029_15_1
    # inputs_1 = tf.layers.max_pooling2d(fuzzy_features, 3, 1, 'same')
    # inputs_2 = tf.layers.max_pooling2d(fuzzy_features, pool_size, 1, 'same')
    # inputs_3 = tf.layers.max_pooling2d(fuzzy_features, 7, 1, 'same')
    # inputs_4 = tf.layers.max_pooling2d(fuzzy_features, 9, 1, 'same')
    # inputs_max_spp = tf.concat([inputs_0,inputs_1,inputs_2,inputs_3,inputs_4], axis=-1)
    # inputs = tf.layers.conv2d(inputs_max_spp, filters, 3, 2, 'same', name=name + '/fuzzy_attention_conv_2')

    outputs = tf.add(tf.multiply(inputs, normal_features), normal_features)
    return inputs, outputs

def global_attention_upsample(low_features, high_features, name='global_attention_upsample_layer' ):

    low_features = tf.layers.conv2d(low_features, low_features.get_shape()[-1], 3, 1, 'same', name=name + '_step1')
    # high_max = tf.reduce_mean(high_features, [1, 2], keep_dims=True)
    high_max = tf.reduce_max(high_features, [1, 2], keep_dims=True)
    high_max = tf.layers.conv2d(high_max, high_features.get_shape()[-1], 1, 1, 'same', name=name + '_step2')
    out = tf.concat([high_features, tf.multiply(low_features, high_max)], axis=-1)

    return out

def transition_down(inputs, filters, filter_size=3, dropout=0.2, train = True, strides=2, name='transition_down'):
    inputs = tf.layers.batch_normalization(inputs, name=name+"/batch_norm")
    inputs = tf.nn.relu(inputs)
    if train:
        inputs = tf.layers.dropout(inputs, dropout)
    inputs = tf.layers.conv2d(inputs, filters, 1, 1, 'same', name=name + '/conv2d_1x1')
    inputs = tf.layers.conv2d(inputs, filters, filter_size, strides, 'same', name=name+'/conv2d_3x3')
    return inputs

def transition_down_max_pool(inputs, filters, filter_size=3, dropout=0.2, train = True, strides=2, name='transition_down'):
    inputs = tf.layers.max_pooling2d(inputs, filter_size, strides, 'same', name=name)
    return inputs

import tensorflow.contrib.crf
def transition_up(inputs, filters, filter_size=3, strides=2):
    inputs = tf.layers.conv2d_transpose(inputs, filters, filter_size, strides, 'same')
    return inputs


def show_all_variables():
    total_count = 0
    for idx, op in enumerate(tf.trainable_variables()):
        shape = op.get_shape()
        count = np.prod(shape)
        print ("[%2d]\t%s\t%s\t=\t%s" % (idx, op.name, shape, count))
        total_count += int(count)
    print("[Total] variable size: %s" % "{:,}".format(total_count))

def l2_loss_all_variables():
    l2_loss = 0
    for idx, op in enumerate(tf.trainable_variables('kernel')):
        l2_loss += tf.nn.l2_loss(op)
    return l2_loss


def summarized_all_variables():
    for idx, op in enumerate(tf.trainable_variables()):
        name = op.name[:op.name.rfind(':')]
        variable_summaries(op, name)

def summarized_special_variables(targetname):
    for idx, op in enumerate(tf.trainable_variables()):
        if op.name in targetname:
            name = op.name[:op.name.rfind(':')]
            variable_summaries(op, name)

def build_loss(output, logits, name):
    softmax_loss = tf.losses.sparse_softmax_cross_entropy(output, logits)
    totle_loss = tf.reduce_mean(softmax_loss)  # sigmoid_loss #
    return totle_loss


def focal_loss(target_tensor, prediction_tensor, name="focal_loss",  weights=None, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    target_tensor = tf.one_hot(target_tensor, 5)
    sigmoid_p = tf.nn.softmax(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    pos_p_sub = array_ops.where(target_tensor >= sigmoid_p, target_tensor - sigmoid_p, zeros)
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    loss = tf.reduce_mean(per_entry_cross_ent)
    tf.add_to_collection("loss",loss)
    return loss

def sparse_weighted_cross_entropy(labels, logits, class_weights = [1, 5, 3, 4, 10], name="loss"):
    # class_weights = [1, 5, 3,
    #                  4, 4]
    # class_weights = [1.4351261795315091, 48.993174617229535, 35.905060084129985,
    #                  47.385762649536716, 44.973769866104227]
    # class_weights = class_weights / 5
    # class_weights[0] = 0
    labels = tf.reshape(labels, [-1])
    logits = tf.reshape(logits, [-1, 5])
    one_hot_label = tf.one_hot(tf.cast(labels, "int32"), 5)

    # batch class weights
    # label_class_sum = tf.reduce_sum(one_hot_label, axis=0)
    # label_class_total_sum = tf.reduce_sum(label_class_sum)
    # class_subtract = tf.subtract(label_class_total_sum, label_class_sum)
    # class_weight = tf.where(tf.not_equal(label_class_sum, 0), class_subtract, tf.zeros_like(class_subtract))
    # class_weight = tf.where(tf.not_equal(class_weight, 0), tf.divide(class_weight, class_weight[0]), tf.zeros_like(class_weight))
    # class_weights = tf.where(tf.not_equal(class_weight[0], 0), class_weight, tf.ones_like(class_weight))
    #class_weights = tf.maximum(tf.divide(class_weight, class_weight[0]), 1e-4)
    #class_weights = [1., 40., 3., 15., 12.]
    #class_weights = [1, 4.48, 51.384, 10.736, 13.36]
    #class_weights = [1., 8., 5., 7.5, 15.]
    # class_weights = [1., 2., 4., 8, 10.]
    # class_weights = [1, 33, 19, 31, 28]
    weights = one_hot_label * class_weights
    weights = tf.reduce_sum(weights, 1)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.cast(labels,"int32"), logits=logits, weights=weights)
    return loss

def dynamic_sparse_weighted_cross_entropy(labels, logits, class_num=5, class_weights = [1, 5, 3, 4, 10], name="loss"):
    # class_weights = [1, 5, 3,
    #                  4, 4]
    # class_weights = [1.4351261795315091, 48.993174617229535, 35.905060084129985,
    #                  47.385762649536716, 44.973769866104227]
    # class_weights = class_weights / 5
    # class_weights[0] = 0
    labels = tf.reshape(labels, [-1])
    logits = tf.reshape(logits, [-1, class_num])
    one_hot_label = tf.one_hot(tf.cast(labels, "int32"), class_num)

    # batch class weights
    # label_class_sum = tf.reduce_sum(one_hot_label, axis=0)
    # label_class_total_sum = tf.reduce_sum(label_class_sum)
    # class_subtract = tf.subtract(label_class_total_sum, label_class_sum)
    # class_weight = tf.where(tf.not_equal(label_class_sum, 0), class_subtract, tf.zeros_like(class_subtract))
    # class_weight = tf.where(tf.not_equal(class_weight, 0), tf.divide(class_weight, class_weight[0]), tf.zeros_like(class_weight))
    # class_weights = tf.where(tf.not_equal(class_weight[0], 0), class_weight, tf.ones_like(class_weight))
    #class_weights = tf.maximum(tf.divide(class_weight, class_weight[0]), 1e-4)
    #class_weights = [1., 40., 3., 15., 12.]
    #class_weights = [1, 4.48, 51.384, 10.736, 13.36]
    #class_weights = [1., 8., 5., 7.5, 15.]
    # class_weights = [1., 2., 4., 8, 10.]
    # class_weights = [1, 33, 19, 31, 28]
    weights = one_hot_label * class_weights
    weights = tf.reduce_sum(weights, 1)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.cast(labels,"int32"), logits=logits, weights=weights)
    return loss

def fuse_sparse_weighted_cross_entropy(labels, logits,class_weights = [1, 5, 3, 4, 10], name="loss"):
    # class_weights = [1, 5, 3,
    #                  4, 4]
    #class_weights = [1.4351261795315091, 48.993174617229535, 35.905060084129985,
                                                  # 47.385762649536716, 44.973769866104227]
    # class_weights = class_weights / 5
    # class_weights[0] = 0
    labels = tf.reshape(labels, [-1])
    logits = tf.reshape(logits, [-1])
    one_hot_label = tf.one_hot(tf.cast(labels,"int32"),5)
    one_hot_logits = tf.one_hot(tf.cast(logits,"int32"),5)
    weights = one_hot_label * class_weights
    weights = tf.reduce_sum(weights, 1)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.cast(labels,"int32"), logits=one_hot_logits, weights=weights)
    return loss

def range_loss(labels, logits, class_weights=[1, 5, 3, 4, 10], name="loss"):
    labels = tf.reshape(labels, [-1])
    logits = tf.reshape(logits, [-1])
    loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)
    # tf.where
    # loss = - tf.div(1.0, tf.log(loss))
    return loss


def variable_summaries(var, name='summaries'):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name):
        # mean = tf.reduce_mean(var)
        # tf.summary.scalar('mean', mean)
        # with tf.name_scope('stddev'):
        #   stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # tf.summary.scalar('stddev', stddev)
        # tf.summary.scalar('max', tf.reduce_max(var))
        # tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def instance_score(inputs, class_num=5, name="instance"):
    positionScore_1 = tf.layers.conv2d(inputs, filters=class_num, kernel_size=(1, 1), strides=(1, 1),
                             name=name + "/positionScore_1", padding='SAME')
    positionScore_2 = tf.layers.conv2d(positionScore_1, filters=class_num, kernel_size=(3, 3), strides=(1, 1),
                             name=name + "/positionScore_2", padding='SAME')
    instance_map_1 = tf.layers.conv2d(inputs, filters=class_num, kernel_size=(3, 3), strides=(1, 1),
                            name=name + "/instance_map_1", padding='SAME')
    instance_map_2 = tf.layers.conv2d(instance_map_1, filters=class_num, kernel_size=(1, 1), strides=(1, 1),
                            name=name + "/instance_map_2", padding='SAME')
    # assume1 = tf.layers.average_pooling2d(instance_map_2, 3, 3, 'same')
    # shape = tf.shape(instance_map_2)
    # assume2 = tf.image.resize_images(assume1, [shape[1], shape[2]])
    return tf.add(instance_map_2, positionScore_2)

def input_crop_layer1(inputs_org, inputs, dimensional='X',train=True):
    shape = tf.shape(inputs_org)
    x = shape[1] // 2
    y = shape[2] // 2
    inputs_crop1 = tf.image.crop_to_bounding_box(inputs_org, offset_height=x // 4, offset_width=y // 4, target_height=x,
                                                 target_width=y)
    inputs_crop2 = tf.image.crop_to_bounding_box(inputs_org, offset_height=x // 4, offset_width=y - (y // 4), target_height=x,
                                                 target_width=y)
    inputs_crop3 = tf.image.crop_to_bounding_box(inputs_org, offset_height=x - (x // 4), offset_width=y // 4, target_height=x,
                                                 target_width=y)
    inputs_crop4 = tf.image.crop_to_bounding_box(inputs_org, offset_height=x - (x // 4), offset_width=y - (y // 4), target_height=x,
                                                 target_width=y)

    # inputs_crop1_2 = tf.image.crop_to_bounding_box(inputs_org, offset_height=x, offset_width=y + (y//2), target_height=x,
    #                                              target_width=y)
    # inputs_crop2_1 = tf.image.crop_to_bounding_box(inputs_org, offset_height=x + (x//2), offset_width=y, target_height=x,
    #                                              target_width=y)
    # inputs_crop2_2 = tf.image.crop_to_bounding_box(inputs_org, offset_height=x + (x//2), offset_width=y + (y//2), target_height=x,
    #                                              target_width=y)
    # inputs_crop2_3 = tf.image.crop_to_bounding_box(inputs_org, offset_height=x + (x//2), offset_width=2*y, target_height=x,
    #                                              target_width=y)
    # inputs_crop3_2 = tf.image.crop_to_bounding_box(inputs_org, offset_height=2*x, offset_width=y + (y//2), target_height=x,
    #                                              target_width=y)

    inputs_crop5 = tf.image.central_crop(inputs_org, 0.5)
    inputs_crop = tf.concat([inputs_crop1, inputs_crop2, inputs_crop3, inputs_crop4, inputs_crop5], -1)
    # inputs_crop = tf.concat([inputs_crop1, inputs_crop2, inputs_crop3, inputs_crop4, inputs_crop5,
    #                          inputs_crop1_2,inputs_crop2_1,inputs_crop2_2,inputs_crop2_3,inputs_crop3_2], -1)
    inputs_crop = tf.layers.batch_normalization(inputs_crop, training=train)
    inputs = tf.concat([inputs, inputs_crop], -1)
    return inputs

def input_crop_layer2(inputs_org, inputs, dimensional='X',train=True):
    shape = tf.shape(inputs_org)
    x = shape[1] // 4
    y = shape[2] // 4
    inputs_crop1 = tf.image.crop_to_bounding_box(inputs_org, offset_height=x, offset_width=y, target_height=x,
                                                 target_width=y)
    inputs_crop2 = tf.image.crop_to_bounding_box(inputs_org, offset_height=x, offset_width=2*y, target_height=x,
                                                 target_width=y)
    inputs_crop3 = tf.image.crop_to_bounding_box(inputs_org, offset_height=2*x, offset_width=y, target_height=x,
                                                 target_width=y)
    inputs_crop4 = tf.image.crop_to_bounding_box(inputs_org, offset_height=2*x, offset_width=2*y, target_height=x,
                                                 target_width=y)

    # inputs_crop1_2 = tf.image.crop_to_bounding_box(inputs_org, offset_height=x, offset_width=y + (y//2), target_height=x,
    #                                              target_width=y)
    # inputs_crop2_1 = tf.image.crop_to_bounding_box(inputs_org, offset_height=x + (x//2), offset_width=y, target_height=x,
    #                                              target_width=y)
    # inputs_crop2_2 = tf.image.crop_to_bounding_box(inputs_org, offset_height=x + (x//2), offset_width=y + (y//2), target_height=x,
    #                                              target_width=y)
    # inputs_crop2_3 = tf.image.crop_to_bounding_box(inputs_org, offset_height=x + (x//2), offset_width=2*y, target_height=x,
    #                                              target_width=y)
    # inputs_crop3_2 = tf.image.crop_to_bounding_box(inputs_org, offset_height=2*x, offset_width=y + (y//2), target_height=x,
    #                                              target_width=y)

    inputs_crop5 = tf.image.central_crop(inputs_org, 0.25)
    inputs_crop = tf.concat([inputs_crop1, inputs_crop2, inputs_crop3, inputs_crop4, inputs_crop5], -1)
    # inputs_crop = tf.concat([inputs_crop1, inputs_crop2, inputs_crop3, inputs_crop4, inputs_crop5,
    #                          inputs_crop1_2,inputs_crop2_1,inputs_crop2_2,inputs_crop2_3,inputs_crop3_2], -1)
    inputs_crop = tf.layers.batch_normalization(inputs_crop, training=train)
    inputs = tf.concat([inputs, inputs_crop], -1)
    return inputs

def concatenate_smaller_images(inputs, to_small_image, train=True):
    shape_input = tf.shape(inputs)
    # BILINEAR
    resize_img = tf.image.resize_images(to_small_image, [shape_input[1], shape_input[2]], method=0)
    # resize_img_2 = tf.image.resize_images(to_small_image, [shape_input[1], shape_input[2]], method=1)
    # resize_img_3 = tf.image.resize_images(to_small_image, [shape_input[1], shape_input[2]], method=2)
    # resize_img_4 = tf.image.resize_images(to_small_image, [shape_input[1], shape_input[2]], method=3)
    # concatenate
    inputs_concatenate = tf.concat([inputs, resize_img], -1)
    #inputs_concatenate = tf.concat([inputs, resize_img, resize_img_2, resize_img_3, resize_img_4], -1)
    return inputs_concatenate

def fuse_results(inputs,filter,name="fuse_out_layer4"):
    inputs = tf.nn.conv2d(inputs, filter, [1, 1, 1, 1], 'SAME', name=name)
    return inputs

def fuse_results_block(inputs, class_num=5, name="fuse"):
    with tf.variable_scope("fuse_out_block"):
        inputs = tf.layers.conv2d(inputs, class_num, 1, 1, 'same', name="restore_the_dimension")
    return inputs

def transform_different_dimension(inputs, dimension=""):
    with tf.variable_scope("transform_different_dimension" + dimension):
        inputs1 = tf.layers.conv2d(inputs, 64, (3, 3), padding='same', activation=tf.nn.leaky_relu, dilation_rate=(1, 2),
                                   name="input1")
        inputs2_1 = tf.layers.conv2d(inputs1, 32, (3, 3), padding='same', activation=tf.nn.leaky_relu, dilation_rate=(3, 3),
                                   name="input2_1")
        inputs3_1 = tf.layers.conv2d(inputs2_1, 32, (3, 3), padding='same', activation=tf.nn.leaky_relu, dilation_rate=(2, 1),
                                   name="input3_1")
        inputs2_2 = tf.layers.conv2d(inputs1, 64, (3, 3), padding='same', activation=tf.nn.leaky_relu, dilation_rate=(2, 4),
                                   name="input2_2")
        inputs2_3 = tf.layers.conv2d(inputs1, 64, (3, 3), padding='same', activation=tf.nn.leaky_relu, dilation_rate=(1, 1),
                                   name="input2_3")
        output = tf.concat([inputs1, inputs2_1, inputs3_1, inputs2_2, inputs2_3], axis=-1)
    return output

def fuse_results_more(input_y, input_z):
    input_y = tf.tile(input_y, multiples=[7, 1, 1, 1])
    input_z = tf.tile(input_z, multiples=[7, 1, 1, 1])
    input_y = input_y[0:15, :, :, :]
    input_z = input_z[0:15, :, :, :]
    return input_y, input_z

def fuse_results_less(input_y, input_z):
    input_y = input_y[0:5, :, :, :]
    input_z = input_z[0:5, :, :, :]
    return input_y, input_z

def results_score(inputs, class_num=5, name="instance"):
    return tf.layers.conv2d(inputs, class_num, 1, 1, 'same', name=name)