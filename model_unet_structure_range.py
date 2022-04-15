from Fuzzy.layers import *
from tensorflow.python.ops import math_ops

class model:
    is_flip = False
    def __init__(self, img, out,  class_num=1, learning_rate=1e-4, train=True, global_step=None):
        self.base_conv_name = 'dilation_conv'
        self.input = img
        self.output = out
        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        # self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 14000, 0.5, staircase=True)
        self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 6000, 0.5, staircase=True)
        self.loss = 0
        self.logits = self.build_net(self.input, class_num, train)
        self.train_op = self.build_optimazer()
        self.result = self.logits

        # correct_prediction = tf.equal(tf.cast(tf.argmax(self.logits, -1), tf.int32), self.output)
        # self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('loss', self.loss)
        show_all_variables()
        #summarized_all_variables()

    # @staticmethod
    def build_net(self, inputs_org, class_num, train=True):
        inputs = self.init_img(inputs_org, train)
        shape = tf.shape(inputs_org)
        self.mid_result = []
        with tf.name_scope("layer1"):
            down_1 = redisual_block(inputs, 3, 64, 3, 0, train, 1, "redisual_layer_1")
            TD_1 = transition_down(down_1, down_1.get_shape()[-1], 3, 0, train, 2, 'transition_down_1')
            #self.mid_result.append(TD_1)

            TD_1 = results_score(TD_1, 32, "pre_instance_1")
            is1 = tf.image.resize_bilinear(TD_1, [shape[1], shape[2]])
            #out1 = results_score(is1, class_num, "instance_1")

            # is1 = results_score(TD_1, class_num, "instance_1")
            # out1 = tf.image.resize_bilinear(is1, [shape[1],shape[2]])
            #self.loss += sparse_weighted_cross_entropy(self.output, out1 ,name="redisual_layer_1/")

        with tf.name_scope("layer2"):

            down_2 = redisual_block(TD_1, 3, 128, 3, 0, train, 1, "redisual_layer_2")
            TD_2 = transition_down(down_2, down_2.get_shape()[-1], 3, 0, train, 2, 'transition_down_2')
            #self.mid_result.append(TD_2)
            TD_2 = results_score(TD_2, 32, "pre_instance_2")
            # is2 =  tf.image.resize_bilinear(TD_2, [shape[1], shape[2]])
            #out2 = results_score(is2, class_num, "instance_2")
            # is2 = results_score(TD_2, class_num, "instance_2")
            # out2 = tf.image.resize_bilinear(is2, [shape[1], shape[2]])
            #self.loss += sparse_weighted_cross_entropy(self.output, out2,name="redisual_layer_2/")

        with tf.name_scope("layer3"):
            down_3 = redisual_block(TD_2, 3, 256, 3, 0, train, 1, "redisual_layer_3")
            TD_3 = transition_down(down_3, down_3.get_shape()[-1], 3, 0, train, 2, 'transition_down_3')
            #self.mid_result.append(TD_3)
            TD_3 = results_score(TD_3, 32, "pre_instance_3")
            # is3 = tf.image.resize_bilinear(TD_3, [shape[1], shape[2]])
            #out3 = results_score(is3, class_num, "instance_3")
            # is3 = results_score(TD_3, class_num, "instance_3")
            # out3 = tf.image.resize_bilinear(is3, [shape[1], shape[2]])
            #self.loss += sparse_weighted_cross_entropy(self.output, out3,name="redisual_layer_3/")

        with tf.name_scope("layer4"):
            down_4 = redisual_block(TD_3, 3, 256, 3, 0, train, 1, "redisual_layer_4")
            #TD_4 = down_4
            #self.mid_result.append(down_4)
            #TD_4 = results_score(TD_4, 32, "pre_instance_4")
            # is4 = tf.image.resize_bilinear(TD_4, [shape[1], shape[2]])
            # out4 = results_score(is4, class_num, "instance_4")

            # is4 = results_score(down_4, class_num, "instance_4")
            # out4 = tf.image.resize_bilinear(is4, [shape[1], shape[2]])
            # self.loss += sparse_weighted_cross_entropy(self.output, out4,name="redisual_layer_4/")

        with tf.name_scope("updown_layer"):
            up1 = tf.layers.conv2d_transpose(inputs=down_4, filters=256, kernel_size=3,
                                           strides=2, padding='same', name='up/conv1')

            #up1 = global_attention_upsample(down_3, up1, name='global_attention_upsample_layer_1')
            #up1_max = tf.reduce_mean(up1, [1,2], name='up/conv1_avg',keep_dims=True)
            #up1 = tf.concat([up1, tf.multiply(down_3,up1_max)], axis=-1, name='up_concat_1' + str(1))

            up1 = tf.concat([up1, down_3], axis=-1, name='up_concat_1' + str(1))
            # up1 = results_score(up1, 32, "pre_instance_5")
            # is5 =  tf.image.resize_bilinear(up1, [shape[1], shape[2]])
            # up1_out = results_score(is5, class_num, "instance_5")
            # is5 = results_score(up1, class_num, "instance_5")
            # up1_out = tf.image.resize_bilinear(is5, [shape[1], shape[2]])
            # self.loss += sparse_weighted_cross_entropy(self.output, up1_out,name="redisual_layer_5/")

            up2 = tf.layers.conv2d_transpose(inputs=up1, filters=128, kernel_size=3, strides=2, padding='same', name='up/conv2')
            # up2 = global_attention_upsample(down_2, up2, name='global_attention_upsample_layer_2')
            # up2_max = tf.reduce_mean(up2, [1,2], name='up/conv2_avg',keep_dims=True)
            # up2 = tf.concat([up2, tf.multiply(down_2,up2_max)], axis=-1, name='up_concat_2' + str(2))
            up2 = tf.concat([up2, down_2], axis=-1,name='up_concat_2' + str(2))
            #up2 = tf.add(up2, down_2, name='up_concat_2' + str(2))
            #up2 = results_score(up2, 32, "pre_instance_6")
            # is6 = tf.image.resize_bilinear(up2, [shape[1], shape[2]])
            # up2_out = results_score(is6, class_num, "instance_6")
            # is6 = results_score(up2, class_num, "instance_6")
            # up2_out = tf.image.resize_bilinear(is6, [shape[1], shape[2]])
            # self.loss += sparse_weighted_cross_entropy(self.output, up2_out,name="redisual_layer_6/")

            up3 = tf.layers.conv2d_transpose(inputs=up2, filters=64, kernel_size=3, strides=2, padding='same', name='up/conv3')
            # up3 = global_attention_upsample(down_1, up3, name='global_attention_upsample_layer_3')

            # up3_max = tf.reduce_mean(up3, [1,2], name='up/conv3_avg',keep_dims=True)
            # up3 = tf.concat([up3, tf.multiply(down_1,up3_max)], axis=-1, name='up_concat_3' + str(3))
            up3 = tf.concat([up3, down_1], axis=-1, name='up_concat_3' + str(3))
            #up3 = tf.add(up3, down_1, name='up_concat_3' + str(3))
            #up3 = results_score(up3, 32, "pre_instance_7")
            # is7 = tf.image.resize_bilinear(up3, [shape[1], shape[2]])
            # up3_out = results_score(is7, class_num, "instance_7")
            # is7 = results_score(up3, class_num, "instance_7")
            # up3_out = tf.image.resize_bilinear(is7, [shape[1], shape[2]])
            # self.loss += sparse_weighted_cross_entropy(self.output, up3_out,name="redisual_layer_7/")

            up4 = tf.layers.conv2d_transpose(up3, filters=32, kernel_size=3, strides=2, padding='same', name='up/conv4')
            # up4 = results_score(up4, 32, "pre_instance_8")
            # up4_out = results_score(up4, class_num, "instance_8")
            # self.loss += sparse_weighted_cross_entropy(self.output, up4_out, name="redisual_layer_8/")

        # with tf.name_scope("layer5"):
        #     down_5 = redisual_block(down_4, 3, 256, 3, 0, train, 4, "redisual_layer_5")
        #     down_5 = down_5 + down_4
        #     is5 = results_score(down_5, class_num, "instance_5")
        #     out5 = tf.image.resize_bilinear(is5, [shape[1], shape[2]])
        #     self.loss += sparse_weighted_cross_entropy(self.output, out5,name="redisual_layer_5/")
        #
        # with tf.name_scope("layer6"):
        #     down_6 = redisual_block(down_5, 3, 512, 3, 0, train, 8, "redisual_layer_6")
        #     out6 = results_score(down_6, class_num, "instance_6")
        #     out6 = tf.image.resize_bilinear(out6, [shape[1],shape[2]])
        #     self.loss += sparse_weighted_cross_entropy(self.output, out6,name="redisual_layer_6/")

        out = up4
        #out = tf.concat([up4_out, out4, out3, out2, out1], -1)
        # out = tf.concat([out4, up1_out, up2_out, up3_out, up4_out], -1)

        #out = tf.concat([up4_out, up3_out, up2_out, up1_out, out4, out3, out2, out1], -1)
        out = results_score(out, class_num, "instance_out_layer4")
        out = tf.squeeze(input=out, axis=-1)
        self.loss += range_loss(self.output, out,name="out/")
        # return tf.nn.softmax(out)
        # return tf.nn.sigmoid(out)
        # 1/(-4ln(x)+1)
        # 1.4x / ((1 + e ^ (-x)))

        return tf.multiply(1.4*out,tf.sigmoid(out))

    @staticmethod
    def init_img(inputs, train=True):

        inputs = tf.layers.batch_normalization(inputs, training=train)
        inputs_conv = tf.layers.conv2d(inputs, 32, 3, 2, 'same', activation=tf.nn.leaky_relu, name='init_img')
        inputs_max = tf.layers.max_pooling2d(inputs, 5, 2, 'same')
        inputs_conv = tf.concat([inputs_conv, inputs_max], -1)
        return inputs_conv

    @staticmethod
    def dense_down(inputs, pool_num, layer_num, grow_filters, filter_size=3, dropout=0.2, train=True):
        DB = inputs
        DB_collect = {}
        for i in range(pool_num):
            with tf.name_scope("dense_down_layers"+str(i+1)):
                filters = DB.get_shape()[-1]
                TD_filters = filters + layer_num*grow_filters
                DB = dense_block(DB, layer_num + i*2, grow_filters, filter_size, dropout, train, up=False)
                DB_collect["DB" +str(i+1)] = DB
                DB = transition_down(DB, TD_filters, 1, 0.2, train, 2)
        return DB, DB_collect

    @staticmethod
    def dense_up(inputs, DB_collect, pool_num, layer_num, grow_filters, filter_size=3, dropout=0.2, train=True):
        DB = inputs
        filters = DB.get_shape()[-1]
        for i in list(range(pool_num))[::-1]:
            with tf.name_scope("dense_up"+ str(i+1)):
                with tf.name_scope("dense_up_layers" + str(i + 1)):
                    TD_filters = filters - (pool_num - i) * grow_filters
                    DB = transition_up(DB, TD_filters, 1, 2)
                    DB = tf.concat([DB_collect["DB" + str(i + 1)], DB], -1)
                    DB = tf.layers.conv2d(DB, TD_filters, 1, 1, 'same')
                    DB = dense_block(DB, layer_num, grow_filters, filter_size, dropout, train, up=True)
        return DB

    def build_optimazer(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        #optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.95, momentum=0.9, epsilon=1e-8)
        # optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.95, momentum=0.9, epsilon=1e-8)
        return optimizer.minimize(self.loss, global_step=self.global_step)

