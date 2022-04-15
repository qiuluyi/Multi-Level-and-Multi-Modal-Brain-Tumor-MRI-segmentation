from Fuzzy.layers import *

class model:
    def __init__(self, img_1, img_2, out_1, out_2, class_num=[2,5], class_weights =[1, 5, 3, 4, 10],
                 learning_rate=1e-4, train=True, global_step=None):
        self.base_conv_name = 'dilation_conv'
        self.input_1 = img_1
        self.input_2 = img_2
        # self.input_2 = tf.concat((img_1, img_2),axis=-1)
        self.output_1 = out_1
        self.output_2 = out_2
        # self.output_2 = tf.concat((out_1, out_2))
        # self.global_step_1 = tf.get_variable('global_step_1', [], initializer=tf.constant_initializer(0), trainable=False)
        # self.global_step_2 = tf.get_variable('global_step_2', [], initializer=tf.constant_initializer(0), trainable=False)
        self.global_step = tf.train.get_or_create_global_step()

        # self.learning_rate_1 = tf.train.exponential_decay(learning_rate, self.global_step_1, 14000, 0.5, staircase=True)
        # self.learning_rate_2 = tf.train.exponential_decay(learning_rate, self.global_step_2, 14000, 0.5, staircase=True)
        self.learning_rate_1 = tf.train.exponential_decay(learning_rate, self.global_step, 14000, 0.5, staircase=True)
        self.learning_rate_2 = tf.train.exponential_decay(learning_rate, self.global_step, 14000, 0.5, staircase=True)
        self.loss = 0
        self.stage = 1
        self.loss_stage1 = 0
        self.loss_stage2 = 0
        #self.logits_1, self.logits_2 = self.build_net(self.input, class_num, class_weights, train)
        self.logits_1 = self.build_net_1(self.input_1, class_num, class_weights, train)
        self.logits_2 = self.build_net_2(self.input_2, class_num, class_weights, train)
        self.train_op_1 = self.build_optimazer(self.learning_rate_1, self.loss_stage1, self.global_step)
        self.train_op_2 = self.build_optimazer(self.learning_rate_2, self.loss_stage2, self.global_step)
        self.result_1 = self.logits_1
        self.result_2 = self.logits_2
        correct_prediction = tf.equal(tf.cast(tf.argmax(self.logits_1, -1), tf.int32), self.output_1)
        self.acc_1 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        correct_prediction = tf.equal(tf.cast(tf.argmax(self.logits_2, -1), tf.int32), self.output_2)
        self.acc_2 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('loss_1', self.loss_stage1)
        tf.summary.scalar('loss_2', self.loss_stage2)
        show_all_variables()

        #summarized_all_variables()

    # @staticmethod
    def build_net_1(self, inputs_org, class_num, class_weights, train=True):
        with tf.variable_scope("stage1"):
            inputs = self.init_img(inputs_org, train)
            shape = tf.shape(inputs_org)
            self.mid_result = []
            with tf.variable_scope("layer1"):
                down_1 = redisual_block(inputs, 3, 64, 3, 0, train, 1, "redisual_layer_1")
                TD_1 = transition_down(down_1, down_1.get_shape()[-1], 3, 0, train, 2, 'transition_down_1')
                #self.mid_result.append(TD_1)
                is1_1 = results_score(TD_1, class_num[0], "instance_1_1")
                out1_1 = tf.image.resize_bilinear(is1_1, [shape[1],shape[2]])
                self.loss_stage1 += dynamic_sparse_weighted_cross_entropy(self.output_1, out1_1, class_num[0], class_weights[0],
                                                                          name="redisual_layer_1_1/")

            with tf.variable_scope("layer2"):
                down_2 = redisual_block(TD_1, 3, 128, 3, 0, train, 1, "redisual_layer_2")
                #TD_2 = transition_down(down_2, down_2.get_shape()[-1], 3, 0, train, 2, 'transition_down_2')

                is2_1 = results_score(down_2, class_num[0], "instance_2_1")
                out2_1 = tf.image.resize_bilinear(is2_1, [shape[1], shape[2]])
                self.loss_stage1 += dynamic_sparse_weighted_cross_entropy(self.output_1, out2_1, class_num[0], class_weights[0],
                                                                          name="redisual_layer_2_1/")

            with tf.variable_scope("updown_layer_stage1"):
                up_1_1 = tf.layers.conv2d_transpose(inputs=down_2, filters=128, kernel_size=3,
                                               strides=2, padding='same', name='up/conv1')
                up_1_1 = tf.concat([up_1_1, down_1], axis=-1, name='up_concat_1' + str(1))
                is_1_5 = results_score(up_1_1, class_num[0], "instance_4")
                up_1_1_out = tf.image.resize_bilinear(is_1_5, [shape[1], shape[2]])
                self.loss_stage1 += dynamic_sparse_weighted_cross_entropy(self.output_1, up_1_1_out,class_num[0],class_weights[0], name="redisual_layer_4/")

                up_1_2 = tf.layers.conv2d_transpose(inputs=up_1_1, filters=64, kernel_size=3, strides=2, padding='same', name='up/conv2')
                is_1_6 = results_score(up_1_2, class_num[0], "instance_5")
                up_1_2_out = tf.image.resize_bilinear(is_1_6, [shape[1], shape[2]])
                self.loss_stage1 += dynamic_sparse_weighted_cross_entropy(self.output_1, up_1_2_out,class_num[0], class_weights[0], name="redisual_layer_5/")

            with tf.variable_scope("instance_layer"):
                out_1 = tf.concat([up_1_2_out, up_1_1_out, out2_1, out1_1], -1)
                out_1 = results_score(out_1, class_num[0], "instance_out_layer4_1")
                self.loss_stage1 += dynamic_sparse_weighted_cross_entropy(self.output_1, out_1, class_num[0], class_weights[0], name="out_1/")
            return tf.nn.softmax(out_1)

    # @staticmethod
    def build_net_2(self, inputs_org, class_num, class_weights, train=True):
        with tf.variable_scope("stage2"):
            inputs = self.init_img(inputs_org, train)
            shape = tf.shape(inputs_org)
            self.mid_result = []
            with tf.variable_scope("layer1"):
                down_1 = redisual_block(inputs, 3, 64, 3, 0, train, 1, "redisual_layer_1")
                TD_1 = transition_down(down_1, down_1.get_shape()[-1], 3, 0, train, 2, 'transition_down_1')
                #self.mid_result.append(TD_1)

                #TD_1 = results_score(TD_1, 32, "pre_instance_1")
                # is1 = tf.image.resize_bilinear(TD_1, [shape[1], shape[2]])
                # out1 = results_score(is1, class_num, "instance_1")

                # is1_2 = results_score(TD_1, class_num[1], "instance_1_2")
                # out1_2 = tf.image.resize_bilinear(is1_2, [shape[1], shape[2]])
                # self.loss_stage2 += dynamic_sparse_weighted_cross_entropy(self.output_2, out1_2, class_num[1], class_weights[1],
                #                                                    name="redisual_layer_1_2/")


            with tf.variable_scope("layer2"):
                down_2 = redisual_block(TD_1, 3, 128, 3, 0, train, 1, "redisual_layer_2")
                TD_2 = transition_down(down_2, down_2.get_shape()[-1], 3, 0, train, 2, 'transition_down_2')
                #self.mid_result.append(TD_2)
                #TD_2 = results_score(TD_2, 32, "pre_instance_2")
                # is2 =  tf.image.resize_bilinear(TD_2, [shape[1], shape[2]])
                # out2 = results_score(is2, class_num, "instance_2")

                # is2_2 = results_score(TD_2, class_num[1], "instance_2_2")
                # out2_2 = tf.image.resize_bilinear(is2_2, [shape[1], shape[2]])
                # self.loss_stage2 += dynamic_sparse_weighted_cross_entropy(self.output_2, out2_2, class_num[1], class_weights[1],
                #                                                    name="redisual_layer_2_2/")

            with tf.variable_scope("layer3"):
                down_3 = redisual_block(TD_2, 3, 256, 3, 0, train, 1, "redisual_layer_3")
                TD_3 = transition_down(down_3, down_3.get_shape()[-1], 3, 0, train, 2, 'transition_down_3')
                #self.mid_result.append(TD_3)
                #TD_3 = results_score(TD_3, 32, "pre_instance_3")
                # is3 = tf.image.resize_bilinear(TD_3, [shape[1], shape[2]])
                # out3 = results_score(is3, class_num, "instance_3")

                # is3_2 = results_score(TD_3, class_num[1], "instance_3_2")
                # out3_2 = tf.image.resize_bilinear(is3_2, [shape[1], shape[2]])
                # self.loss_stage2 += dynamic_sparse_weighted_cross_entropy(self.output_2, out3_2, class_num[1], class_weights[1],
                #                                                    name="redisual_layer_3_2/")

            with tf.variable_scope("layer4"):
                down_4 = redisual_block(TD_3, 3, 256, 3, 0, train, 1, "redisual_layer_4")
                #TD_4 = down_4
                #self.mid_result.append(down_4)
                #TD_4 = results_score(TD_4, 32, "pre_instance_4")
                # is4 = tf.image.resize_bilinear(TD_4, [shape[1], shape[2]])
                # out4 = results_score(is4, class_num, "instance_4")
                is4 = results_score(down_4, class_num[1], "instance_4")
                out4 = tf.image.resize_bilinear(is4, [shape[1], shape[2]])
                self.loss_stage2 += dynamic_sparse_weighted_cross_entropy(self.output_2, out4, class_num[1], class_weights[1],
                                                                          name="redisual_layer_4/")

            with tf.variable_scope("updown_layer_stage2"):
                up_2_1 = tf.layers.conv2d_transpose(inputs=down_4, filters=256, kernel_size=3,
                                               strides=2, padding='same', name='up/conv1')
                up_2_1 = tf.concat([up_2_1, down_3], axis=-1, name='up_concat_1' + str(1))
                is_2_5 = results_score(up_2_1, class_num[1], "instance_5")
                up_2_1_out = tf.image.resize_bilinear(is_2_5, [shape[1], shape[2]])
                self.loss_stage2 += dynamic_sparse_weighted_cross_entropy(self.output_2, up_2_1_out, class_num[1], class_weights[1], name="redisual_layer_5/")

                up_2_2 = tf.layers.conv2d_transpose(inputs=up_2_1, filters=128, kernel_size=3, strides=2, padding='same', name='up/conv2')
                up_2_2 = tf.concat([up_2_2, down_2], axis=-1,name='up_concat_2' + str(2))
                is_2_6 = results_score(up_2_2, class_num[1], "instance_6")
                up_2_2_out = tf.image.resize_bilinear(is_2_6, [shape[1], shape[2]])
                self.loss_stage2 += dynamic_sparse_weighted_cross_entropy(self.output_2, up_2_2_out, class_num[1], class_weights[1], name="redisual_layer_6/")

                up_2_3 = tf.layers.conv2d_transpose(inputs=up_2_2, filters=64, kernel_size=3, strides=2, padding='same', name='up/conv3')
                up_2_3 = tf.concat([up_2_3, down_1], axis=-1, name='up_concat_3' + str(3))
                is_2_7 = results_score(up_2_3, class_num[1], "instance_7")
                up_2_3_out = tf.image.resize_bilinear(is_2_7, [shape[1], shape[2]])
                self.loss_stage2 += dynamic_sparse_weighted_cross_entropy(self.output_2, up_2_3_out, class_num[1], class_weights[1], name="redisual_layer_7/")

                up_2_4 = tf.layers.conv2d_transpose(up_2_3, filters=32, kernel_size=3, strides=2, padding='same', name='up/conv4')
                up_2_4_out = results_score(up_2_4, class_num[1], "instance_8")
                self.loss_stage2 += dynamic_sparse_weighted_cross_entropy(self.output_2, up_2_4_out, class_num[1], class_weights[1], name="redisual_layer_8/")

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

            # out = up4_out
            with tf.variable_scope("instance_layer"):
                out_2 = tf.concat([out4, up_2_1_out, up_2_2_out, up_2_3_out, up_2_4_out], -1)
                #out = tf.concat([up4_out, up3_out, up2_out, up1_out, out4, out3, out2, out1], -1)
                out_2 = results_score(out_2, class_num[1], "instance_out_layer4_2")
                self.loss_stage2 += dynamic_sparse_weighted_cross_entropy(self.output_2, out_2, class_num[1], class_weights[1], name="out_2/")
            return tf.nn.softmax(out_2)

    # @staticmethod
    def build_net(self, inputs_org, class_num, class_weights, train=True):
        inputs = self.init_img(inputs_org, train)
        shape = tf.shape(inputs_org)
        self.mid_result = []
        with tf.variable_scope("layer1"):
            down_1 = redisual_block(inputs, 3, 64, 3, 0, train, 1, "redisual_layer_1")
            TD_1 = transition_down(down_1, down_1.get_shape()[-1], 3, 0, train, 2, 'transition_down_1')
            #self.mid_result.append(TD_1)

            #TD_1 = results_score(TD_1, 32, "pre_instance_1")
            # is1 = tf.image.resize_bilinear(TD_1, [shape[1], shape[2]])
            # out1 = results_score(is1, class_num, "instance_1")

            is1_1 = results_score(TD_1, class_num[0], "instance_1_1")
            out1_1 = tf.image.resize_bilinear(is1_1, [shape[1],shape[2]])
            self.loss_stage1 += dynamic_sparse_weighted_cross_entropy(self.output_1, out1_1, class_num[0], class_weights[0],
                                                                      name="redisual_layer_1_1/")

            is1_2 = results_score(TD_1, class_num[1], "instance_1_2")
            out1_2 = tf.image.resize_bilinear(is1_2, [shape[1], shape[2]])
            self.loss_stage2 += dynamic_sparse_weighted_cross_entropy(self.output_2, out1_2, class_num[1], class_weights[1],
                                                               name="redisual_layer_1_2/")


        with tf.variable_scope("layer2"):
            down_2 = redisual_block(TD_1, 3, 128, 3, 0, train, 1, "redisual_layer_2")
            TD_2 = transition_down(down_2, down_2.get_shape()[-1], 3, 0, train, 2, 'transition_down_2')
            #self.mid_result.append(TD_2)
            #TD_2 = results_score(TD_2, 32, "pre_instance_2")
            # is2 =  tf.image.resize_bilinear(TD_2, [shape[1], shape[2]])
            # out2 = results_score(is2, class_num, "instance_2")
            is2_1 = results_score(TD_2, class_num[0], "instance_2_1")
            out2_1 = tf.image.resize_bilinear(is2_1, [shape[1], shape[2]])
            self.loss_stage1 += dynamic_sparse_weighted_cross_entropy(self.output_1, out2_1, class_num[0], class_weights[0],
                                                                      name="redisual_layer_2_1/")

            is2_2 = results_score(TD_2, class_num[1], "instance_2_2")
            out2_2 = tf.image.resize_bilinear(is2_2, [shape[1], shape[2]])
            self.loss_stage2 += dynamic_sparse_weighted_cross_entropy(self.output_2, out2_2, class_num[1], class_weights[1],
                                                               name="redisual_layer_2_2/")

        with tf.variable_scope("layer3"):
            down_3 = redisual_block(TD_2, 3, 256, 3, 0, train, 1, "redisual_layer_3")
            TD_3 = transition_down(down_3, down_3.get_shape()[-1], 3, 0, train, 2, 'transition_down_3')
            #self.mid_result.append(TD_3)
            #TD_3 = results_score(TD_3, 32, "pre_instance_3")
            # is3 = tf.image.resize_bilinear(TD_3, [shape[1], shape[2]])
            # out3 = results_score(is3, class_num, "instance_3")
            is3_1 = results_score(TD_3, class_num[0], "instance_3_1")
            out3_1 = tf.image.resize_bilinear(is3_1, [shape[1], shape[2]])
            self.loss_stage1 += dynamic_sparse_weighted_cross_entropy(self.output_1, out3_1, class_num[0], class_weights[0],
                                                                      name="redisual_layer_3_1/")

            is3_2 = results_score(TD_3, class_num[1], "instance_3_2")
            out3_2 = tf.image.resize_bilinear(is3_2, [shape[1], shape[2]])
            self.loss_stage2 += dynamic_sparse_weighted_cross_entropy(self.output_2, out3_2, class_num[1], class_weights[1],
                                                               name="redisual_layer_3_2/")

        with tf.variable_scope("updown_layer_stage1"):
            up_1_1 = tf.layers.conv2d_transpose(inputs=down_3, filters=128, kernel_size=3,
                                           strides=2, padding='same', name='up/conv1')
            up_1_1 = tf.concat([up_1_1, down_2], axis=-1, name='up_concat_1' + str(1))
            is_1_5 = results_score(up_1_1, class_num[0], "instance_4")
            up_1_1_out = tf.image.resize_bilinear(is_1_5, [shape[1], shape[2]])
            self.loss_stage1 += dynamic_sparse_weighted_cross_entropy(self.output_1, up_1_1_out,class_num[0],class_weights[0], name="redisual_layer_4/")

            up_1_2 = tf.layers.conv2d_transpose(inputs=up_1_1, filters=64, kernel_size=3, strides=2, padding='same', name='up/conv2')
            up_1_2 = tf.concat([up_1_2, down_1], axis=-1,name='up_concat_2' + str(2))
            is_1_6 = results_score(up_1_2, class_num[0], "instance_5")
            up_1_2_out = tf.image.resize_bilinear(is_1_6, [shape[1], shape[2]])
            self.loss_stage1 += dynamic_sparse_weighted_cross_entropy(self.output_1, up_1_2_out,class_num[0], class_weights[0], name="redisual_layer_5/")

            up_1_3 = tf.layers.conv2d_transpose(inputs=up_1_2, filters=32, kernel_size=3, strides=2, padding='same', name='up/conv3')
            is_1_7 = results_score(up_1_3, class_num[0], "instance_7")
            up_1_3_out = tf.image.resize_bilinear(is_1_7, [shape[1], shape[2]])
            self.loss_stage1 += dynamic_sparse_weighted_cross_entropy(self.output_1, up_1_3_out, class_num[0], class_weights[0], name="redisual_layer_6/")

        with tf.variable_scope("layer4"):
            down_4 = redisual_block(TD_3, 3, 256, 3, 0, train, 1, "redisual_layer_4")
            #TD_4 = down_4
            #self.mid_result.append(down_4)
            #TD_4 = results_score(TD_4, 32, "pre_instance_4")
            # is4 = tf.image.resize_bilinear(TD_4, [shape[1], shape[2]])
            # out4 = results_score(is4, class_num, "instance_4")
            is4 = results_score(down_4, class_num[1], "instance_4")
            out4 = tf.image.resize_bilinear(is4, [shape[1], shape[2]])
            self.loss_stage2 += dynamic_sparse_weighted_cross_entropy(self.output_2, out4, class_num[1], class_weights[1],
                                                                      name="redisual_layer_4/")

        with tf.variable_scope("updown_layer_stage2"):
            up_2_1 = tf.layers.conv2d_transpose(inputs=down_4, filters=256, kernel_size=3,
                                           strides=2, padding='same', name='up/conv1')
            up_2_1 = tf.concat([up_2_1, down_3], axis=-1, name='up_concat_1' + str(1))
            is_2_5 = results_score(up_2_1, class_num[1], "instance_5")
            up_2_1_out = tf.image.resize_bilinear(is_2_5, [shape[1], shape[2]])
            self.loss_stage2 += dynamic_sparse_weighted_cross_entropy(self.output_2, up_2_1_out, class_num[1], class_weights[1], name="redisual_layer_5/")

            up_2_2 = tf.layers.conv2d_transpose(inputs=up_2_1, filters=128, kernel_size=3, strides=2, padding='same', name='up/conv2')
            up_2_2 = tf.concat([up_2_2, down_2], axis=-1,name='up_concat_2' + str(2))
            is_2_6 = results_score(up_2_2, class_num[1], "instance_6")
            up_2_2_out = tf.image.resize_bilinear(is_2_6, [shape[1], shape[2]])
            self.loss_stage2 += dynamic_sparse_weighted_cross_entropy(self.output_2, up_2_2_out, class_num[1], class_weights[1], name="redisual_layer_6/")

            up_2_3 = tf.layers.conv2d_transpose(inputs=up_2_2, filters=64, kernel_size=3, strides=2, padding='same', name='up/conv3')
            up_2_3 = tf.concat([up_2_3, down_1], axis=-1, name='up_concat_3' + str(3))
            is_2_7 = results_score(up_2_3, class_num[1], "instance_7")
            up_2_3_out = tf.image.resize_bilinear(is_2_7, [shape[1], shape[2]])
            self.loss_stage2 += dynamic_sparse_weighted_cross_entropy(self.output_2, up_2_3_out, class_num[1], class_weights[1], name="redisual_layer_7/")

            up_2_4 = tf.layers.conv2d_transpose(up_2_3, filters=32, kernel_size=3, strides=2, padding='same', name='up/conv4')
            up_2_4_out = results_score(up_2_4, class_num[1], "instance_8")
            self.loss_stage2 += dynamic_sparse_weighted_cross_entropy(self.output_2, up_2_4_out, class_num[1], class_weights[1], name="redisual_layer_8/")

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

        # out = up4_out
        with tf.variable_scope("instance_layer"):
            out_1 = tf.concat([up_1_3_out, up_1_2_out, up_1_1_out, out3_1, out2_1, out1_1], -1)
            out_1 = results_score(out_1, class_num[0], "instance_out_layer4_1")
            self.loss_stage1 += dynamic_sparse_weighted_cross_entropy(self.output_1, out_1, class_num[0], class_weights[0], name="out_1/")

            out_2 = tf.concat([out4, up_2_1_out, up_2_2_out, up_2_3_out, up_2_4_out], -1)
            #out = tf.concat([up4_out, up3_out, up2_out, up1_out, out4, out3, out2, out1], -1)
            out_2 = results_score(out_2, class_num[1], "instance_out_layer4_2")
            self.loss_stage2 += dynamic_sparse_weighted_cross_entropy(self.output_2, out_2, class_num[1], class_weights[1], name="out_2/")
        return tf.nn.softmax(out_1), tf.nn.softmax(out_2)

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

    def build_optimazer(self, learning_rate, loss, global_step):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        #optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.95, momentum=0.9, epsilon=1e-8)
        # optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.95, momentum=0.9, epsilon=1e-8)
        return optimizer.minimize(loss, global_step=global_step)

