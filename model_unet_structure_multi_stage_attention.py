from Fuzzy.layers import *
from Fuzzy.attention import atblock

class model:
    def __init__(self, img, out, class_num=[2,5], class_weights =[1, 5, 3, 4, 10],
                 learning_rate=1e-4, train=True, global_step=None):
        self.base_conv_name = 'dilation_conv'
        self.input = img
        self.output = out
        # self.global_step_1 = tf.get_variable('global_step_1', [], initializer=tf.constant_initializer(0), trainable=False)
        # self.global_step_2 = tf.get_variable('global_step_2', [], initializer=tf.constant_initializer(0), trainable=False)
        self.global_step = tf.train.get_or_create_global_step()

        # self.learning_rate_1 = tf.train.exponential_decay(learning_rate, self.global_step_1, 14000, 0.5, staircase=True)
        # self.learning_rate_2 = tf.train.exponential_decay(learning_rate, self.global_step_2, 14000, 0.5, staircase=True)
        self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 14000, 0.5, staircase=True)
        self.loss = 0
        self.stage = 1
        #self.logits_1, self.logits_2 = self.build_net(self.input, class_num, class_weights, train)
        # self.logits = self.build_net(self.input, class_num, class_weights, train)
        #self.logits = self.build_net_new(self.input, class_num, class_weights, train)
        #self.logits = self.build_parallel_pyramid_net(self.input, class_num, class_weights, train)
        #self.logits = self.build_cross_parallel_pyramid_net(self.input, class_num, class_weights, train)
        self.logits = self.build_multi_parallel_pyramid_net(self.input, class_num, class_weights, train)
        #self.logits = self.build_multi_parallel_pyramid_attention_net(self.input, class_num, class_weights, train)

        self.train_op = self.build_optimazer(self.learning_rate, self.loss, self.global_step)
        self.result = self.logits
        correct_prediction = tf.equal(tf.cast(tf.argmax(self.logits, -1), tf.int32), self.output)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('loss', self.loss)
        show_all_variables()

        #summarized_all_variables()


    # @staticmethod
    def build_net(self, inputs_org, class_num, class_weights, train=True):
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
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, out4, class_num[1], class_weights[1],
                                                                          name="redisual_layer_4/")

            with tf.variable_scope("updown_layer_stage2"):
                up_2_1 = tf.layers.conv2d_transpose(inputs=down_4, filters=256, kernel_size=3,
                                               strides=2, padding='same', name='up/conv1')
                up_2_1 = tf.concat([up_2_1, down_3], axis=-1, name='up_concat_1' + str(1))
                is_2_5 = results_score(up_2_1, class_num[1], "instance_5")
                up_2_1_out = tf.image.resize_bilinear(is_2_5, [shape[1], shape[2]])
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_1_out, class_num[1], class_weights[1], name="redisual_layer_5/")

                up_2_2 = tf.layers.conv2d_transpose(inputs=up_2_1, filters=128, kernel_size=3, strides=2, padding='same', name='up/conv2')
                up_2_2 = tf.concat([up_2_2, down_2], axis=-1,name='up_concat_2' + str(2))
                is_2_6 = results_score(up_2_2, class_num[1], "instance_6")
                up_2_2_out = tf.image.resize_bilinear(is_2_6, [shape[1], shape[2]])
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_2_out, class_num[1], class_weights[1], name="redisual_layer_6/")

                up_2_3 = tf.layers.conv2d_transpose(inputs=up_2_2, filters=64, kernel_size=3, strides=2, padding='same', name='up/conv3')
                up_2_3 = tf.concat([up_2_3, down_1], axis=-1, name='up_concat_3' + str(3))
                is_2_7 = results_score(up_2_3, class_num[1], "instance_7")
                up_2_3_out = tf.image.resize_bilinear(is_2_7, [shape[1], shape[2]])
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_3_out, class_num[1], class_weights[1], name="redisual_layer_7/")

                up_2_4 = tf.layers.conv2d_transpose(up_2_3, filters=32, kernel_size=3, strides=2, padding='same', name='up/conv4')
                up_2_4_out = results_score(up_2_4, class_num[1], "instance_8")
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_4_out, class_num[1], class_weights[1], name="redisual_layer_8/")

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
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, out_2, class_num[1], class_weights[1], name="out_2/")
            return tf.nn.softmax(out_2)

    def build_parallel_pyramid_net(self, inputs_org, class_num, class_weights, train=True):
        with tf.variable_scope("stage2"):
            inputs = self.init_img(inputs_org, train)
            shape = tf.shape(inputs_org)
            self.mid_result = []
            with tf.variable_scope("layer1"):
                down_1 = redisual_block(inputs, 3, 64, 3, 0, train, 1, "redisual_layer_1")
                TD_1 = transition_down(down_1, down_1.get_shape()[-1], 3, 0, train, 2, 'transition_down_1')
                # parallel
                parallel_1 = redisual_block(down_1, 3, 64, 3, 0, train, 1, "parallel_redisual_layer_1")
                p_TD_1 = transition_down(parallel_1, parallel_1.get_shape()[-1], 3, 0, train, 2, 'parallel_transition_down_1')
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
                # parallel
                p_TD_1 = tf.concat([down_2, p_TD_1], axis=-1)
                parallel_2 = redisual_block(p_TD_1, 3, 128, 3, 0, train, 1, "parallel_redisual_layer_2")
                p_TD_2 = transition_down(parallel_2, parallel_2.get_shape()[-1], 3, 0, train, 2, 'parallel_transition_down_2')
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

                # parallel
                p_TD_2 = tf.concat([down_3, p_TD_2], axis=-1)
                parallel_3 = redisual_block(p_TD_2, 3, 256, 3, 0, train, 1, "parallel_redisual_layer_3")
                p_TD_3 = transition_down(parallel_3, parallel_3.get_shape()[-1], 3, 0, train, 2, 'parallel_transition_down_3')

                #self.mid_result.append(TD_3)
                #TD_3 = results_score(TD_3, 32, "pre_instance_3")
                # is3 = tf.image.resize_bilinear(TD_3, [shape[1], shape[2]])
                # out3 = results_score(is3, class_num, "instance_3")

                # is3_2 = results_score(TD_3, class_num[1], "instance_3_2")
                # out3_2 = tf.image.resize_bilinear(is3_2, [shape[1], shape[2]])
                # self.loss_stage2 += dynamic_sparse_weighted_cross_entropy(self.output_2, out3_2, class_num[1], class_weights[1],
                #                                                    name="redisual_layer_3_2/")

            with tf.variable_scope("layer4"):

                down_4 = tf.concat([TD_3, p_TD_3], axis=-1)
                down_4 = redisual_block(down_4, 3, 256, 3, 0, train, 1, "redisual_layer_4")

                #down_4 = redisual_block(TD_3, 3, 256, 3, 0, train, 1, "redisual_layer_4")

                # parallel
                #p_TD_3 = tf.concat([down_4, p_TD_3], axis=-1)
                #parallel_4 = redisual_block(p_TD_3, 3, 256, 3, 0, train, 1, "parallel_redisual_layer_4")

                #p_TD_4 = transition_down(parallel_4, parallel_4.get_shape()[-1], 3, 0, train, 2, 'parallel_transition_down_4')

                #TD_4 = down_4
                #self.mid_result.append(down_4)
                #TD_4 = results_score(TD_4, 32, "pre_instance_4")
                # is4 = tf.image.resize_bilinear(TD_4, [shape[1], shape[2]])
                # out4 = results_score(is4, class_num, "instance_4")
                # is4 = results_score(down_4, class_num[1], "instance_4")
                # out4 = tf.image.resize_bilinear(is4, [shape[1], shape[2]])
                # self.loss += dynamic_sparse_weighted_cross_entropy(self.output, out4, class_num[1], class_weights[1],
                #                                                           name="redisual_layer_4/")

                is4 = results_score(down_4, class_num[1], "instance_4")
                out4 = tf.image.resize_bilinear(is4, [shape[1], shape[2]])
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, out4, class_num[1], class_weights[1],
                                                                          name="redisual_layer_4/")

            with tf.variable_scope("updown_layer_stage2"):
                up_2_1 = tf.layers.conv2d_transpose(inputs=down_4, filters=256, kernel_size=3,
                                               strides=2, padding='same', name='up/conv1')
                up_2_1 = tf.concat([up_2_1, parallel_3], axis=-1, name='up_concat_1' + str(1))
                is_2_5 = results_score(up_2_1, class_num[1], "instance_5")
                up_2_1_out = tf.image.resize_bilinear(is_2_5, [shape[1], shape[2]])
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_1_out, class_num[1], class_weights[1], name="redisual_layer_5/")

                up_2_2 = tf.layers.conv2d_transpose(inputs=up_2_1, filters=128, kernel_size=3, strides=2, padding='same', name='up/conv2')
                up_2_2 = tf.concat([up_2_2, parallel_2], axis=-1,name='up_concat_2' + str(2))
                is_2_6 = results_score(up_2_2, class_num[1], "instance_6")
                up_2_2_out = tf.image.resize_bilinear(is_2_6, [shape[1], shape[2]])
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_2_out, class_num[1], class_weights[1], name="redisual_layer_6/")

                up_2_3 = tf.layers.conv2d_transpose(inputs=up_2_2, filters=64, kernel_size=3, strides=2, padding='same', name='up/conv3')
                up_2_3 = tf.concat([up_2_3, parallel_1], axis=-1, name='up_concat_3' + str(3))
                is_2_7 = results_score(up_2_3, class_num[1], "instance_7")
                up_2_3_out = tf.image.resize_bilinear(is_2_7, [shape[1], shape[2]])
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_3_out, class_num[1], class_weights[1], name="redisual_layer_7/")

                up_2_4 = tf.layers.conv2d_transpose(up_2_3, filters=32, kernel_size=3, strides=2, padding='same', name='up/conv4')
                up_2_4_out = results_score(up_2_4, class_num[1], "instance_8")
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_4_out, class_num[1], class_weights[1], name="redisual_layer_8/")

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
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, out_2, class_num[1], class_weights[1], name="out_2/")
            return tf.nn.softmax(out_2)

    def build_multi_parallel_pyramid_net(self, inputs_org, class_num, class_weights, train=True):
        with tf.variable_scope("stage2"):
            inputs = self.init_img(inputs_org, train)
            shape = tf.shape(inputs_org)
            self.mid_result = []
            with tf.variable_scope("layer1"):

                # inputs = concatenate_smaller_images(inputs, inputs_org)
                # # crop
                # inputs = input_crop_layer1(inputs_org, inputs, "X", train)

                down_1 = redisual_block(inputs, 3, 64, 3, 0, train, 1, "redisual_layer_1")
                TD_1 = transition_down(down_1, down_1.get_shape()[-1], 3, 0, train, 2, 'transition_down_1')
                # parallel
                #parallel_1_1 = redisual_block(down_1, 3, 64, 3, 0, train, 1, "parallel_redisual_layer_1_1")
                #p_TD_1_1 = transition_down(parallel_1_1, parallel_1_1.get_shape()[-1], 3, 0, train, 2, 'parallel_transition_down_1_1')

                #parallel_2_1 = redisual_block(parallel_1_1, 3, 64, 3, 0, train, 1, "parallel_redisual_layer_2_1")
                #p_TD_2_1 = transition_down(parallel_2_1, parallel_2_1.get_shape()[-1], 3, 0, train, 2,  'parallel_transition_down_2_1')

                # parallel_3_1 = redisual_block(parallel_2_1, 3, 64, 3, 0, train, 1, "parallel_redisual_layer_3_1")
                # p_TD_3_1 = transition_down(parallel_3_1, parallel_3_1.get_shape()[-1], 3, 0, train, 2,
                #                            'parallel_transition_down_3_1')

                #self.mid_result.append(TD_1)

                #TD_1 = results_score(TD_1, 32, "pre_instance_1")
                # is1 = tf.image.resize_bilinear(TD_1, [shape[1], shape[2]])
                # out1 = results_score(is1, class_num, "instance_1")

                # is1_2 = results_score(TD_1, class_num[1], "instance_1_2")
                # out1_2 = tf.image.resize_bilinear(is1_2, [shape[1], shape[2]])
                # self.loss_stage2 += dynamic_sparse_weighted_cross_entropy(self.output_2, out1_2, class_num[1], class_weights[1],
                #                                                    name="redisual_layer_1_2/")


            with tf.variable_scope("layer2"):

                # crop
                # TD_1 = input_crop_layer2(inputs_org, TD_1, "X", train)
                # p_TD_1_1 = input_crop_layer2(inputs_org, p_TD_1_1, "X", train)
                # p_TD_2_1 = input_crop_layer2(inputs_org, p_TD_2_1, "X", train)

                down_2 = redisual_block(TD_1, 3, 128, 3, 0, train, 1, "redisual_layer_2")
                TD_2 = transition_down(down_2, down_2.get_shape()[-1], 3, 0, train, 2, 'transition_down_2')
                # parallel
                #p_TD_1_1 = tf.concat([down_2, p_TD_1_1], axis=-1)
                #parallel_1_2 = redisual_block(p_TD_1_1, 3, 128, 3, 0, train, 1, "parallel_redisual_layer_1_2")
                parallel_1_2 = redisual_block(TD_1, 3, 128, 3, 0, train, 1, "parallel_redisual_layer_1_2")
                p_TD_1_2 = transition_down(parallel_1_2, parallel_1_2.get_shape()[-1], 3, 0, train, 2, 'parallel_transition_down_1_2')

                #p_TD_2_1 = tf.concat([parallel_1_2, p_TD_2_1], axis=-1)
                #parallel_2_2 = redisual_block(p_TD_2_1, 3, 128, 3, 0, train, 1, "parallel_redisual_layer_2_2")
                parallel_2_2 = redisual_block(TD_1, 3, 128, 3, 0, train, 1, "parallel_redisual_layer_2_2")
                p_TD_2_2 = transition_down(parallel_2_2, parallel_2_2.get_shape()[-1], 3, 0, train, 2,
                                           'parallel_transition_down_2_2')

                # p_TD_3_1 = tf.concat([parallel_2_2, p_TD_3_1], axis=-1)
                # parallel_3_2 = redisual_block(p_TD_3_1, 3, 128, 3, 0, train, 1, "parallel_redisual_layer_3_2")
                # p_TD_3_2 = transition_down(parallel_3_2, parallel_3_2.get_shape()[-1], 3, 0, train, 2,
                #                            'parallel_transition_down_3_2')

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

                # parallel
                p_TD_1_2 = tf.concat([down_3, p_TD_1_2], axis=-1)
                parallel_1_3 = redisual_block(p_TD_1_2, 3, 256, 3, 0, train, 1, "parallel_redisual_layer_1_3")
                p_TD_1_3 = transition_down(parallel_1_3, parallel_1_3.get_shape()[-1], 3, 0, train, 2, 'parallel_transition_down_1_3')

                p_TD_2_2 = tf.concat([parallel_1_3, p_TD_2_2], axis=-1)
                parallel_2_3 = redisual_block(p_TD_2_2, 3, 256, 3, 0, train, 1, "parallel_redisual_layer_2_3")
                p_TD_2_3 = transition_down(parallel_2_3, parallel_2_3.get_shape()[-1], 3, 0, train, 2, 'parallel_transition_down_2_3')

                # p_TD_3_2 = tf.concat([parallel_2_3, p_TD_3_2], axis=-1)
                # parallel_3_3 = redisual_block(p_TD_3_2, 3, 256, 3, 0, train, 1, "parallel_redisual_layer_3_3")
                # p_TD_3_3 = transition_down(parallel_3_3, parallel_3_3.get_shape()[-1], 3, 0, train, 2, 'parallel_transition_down_3_3')

                #self.mid_result.append(TD_3)
                #TD_3 = results_score(TD_3, 32, "pre_instance_3")
                # is3 = tf.image.resize_bilinear(TD_3, [shape[1], shape[2]])
                # out3 = results_score(is3, class_num, "instance_3")

                # is3_2 = results_score(TD_3, class_num[1], "instance_3_2")
                # out3_2 = tf.image.resize_bilinear(is3_2, [shape[1], shape[2]])
                # self.loss_stage2 += dynamic_sparse_weighted_cross_entropy(self.output_2, out3_2, class_num[1], class_weights[1],
                #                                                    name="redisual_layer_3_2/")

            with tf.variable_scope("layer4"):

                #down_4 = redisual_block(p_TD_2_3, 3, 256, 3, 0, train, 1, "redisual_layer_4")

                down_4 = tf.concat([TD_3, p_TD_1_3, p_TD_2_3], axis=-1)
                # down_4 = tf.concat([TD_3, p_TD_1_3, p_TD_2_3, p_TD_3_3], axis=-1)
                down_4 = redisual_block(down_4, 3, 256, 3, 0, train, 1, "redisual_layer_4")

                #TD_4 = down_4
                #self.mid_result.append(down_4)
                #TD_4 = results_score(TD_4, 32, "pre_instance_4")
                # is4 = tf.image.resize_bilinear(TD_4, [shape[1], shape[2]])
                # out4 = results_score(is4, class_num, "instance_4")
                # is4 = results_score(down_4, class_num[1], "instance_4")
                # out4 = tf.image.resize_bilinear(is4, [shape[1], shape[2]])
                # self.loss += dynamic_sparse_weighted_cross_entropy(self.output, out4, class_num[1], class_weights[1],
                #                                                           name="redisual_layer_4/")

                is4 = results_score(down_4, class_num[1], "instance_4")
                out4 = tf.image.resize_bilinear(is4, [shape[1], shape[2]])
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, out4, class_num[1], class_weights[1],
                                                                          name="redisual_layer_4/")
            # U up conv
            with tf.variable_scope("updown_layer_stage2"):
                up_2_1 = tf.layers.conv2d_transpose(inputs=down_4, filters=256, kernel_size=3,
                                               strides=2, padding='same', name='up/conv1')
                up_2_1 = tf.concat([up_2_1, parallel_2_3], axis=-1, name='up_concat_1' + str(1))
                is_2_5 = results_score(up_2_1, class_num[1], "instance_5")
                up_2_1_out = tf.image.resize_bilinear(is_2_5, [shape[1], shape[2]])
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_1_out, class_num[1], class_weights[1], name="redisual_layer_5/")

                up_2_2 = tf.layers.conv2d_transpose(inputs=up_2_1, filters=128, kernel_size=3, strides=2, padding='same', name='up/conv2')
                up_2_2 = tf.concat([up_2_2, parallel_2_2], axis=-1,name='up_concat_2' + str(2))
                is_2_6 = results_score(up_2_2, class_num[1], "instance_6")
                up_2_2_out = tf.image.resize_bilinear(is_2_6, [shape[1], shape[2]])
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_2_out, class_num[1], class_weights[1], name="redisual_layer_6/")

                up_2_3 = tf.layers.conv2d_transpose(inputs=up_2_2, filters=64, kernel_size=3, strides=2, padding='same', name='up/conv3')
                #up_2_3 = tf.concat([up_2_3, parallel_2_1], axis=-1, name='up_concat_3' + str(3))
                is_2_7 = results_score(up_2_3, class_num[1], "instance_7")
                up_2_3_out = tf.image.resize_bilinear(is_2_7, [shape[1], shape[2]])
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_3_out, class_num[1], class_weights[1], name="redisual_layer_7/")

                up_2_4 = tf.layers.conv2d_transpose(up_2_3, filters=32, kernel_size=3, strides=2, padding='same', name='up/conv4')
                up_2_4_out = results_score(up_2_4, class_num[1], "instance_8")
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_4_out, class_num[1], class_weights[1], name="redisual_layer_8/")

            # multi U up conv
            # with tf.variable_scope("updown_layer_stage2"):
            #
            #     # # method 1
            #     # # 2^(k+1) + 1  2^k
            #     # parallel_2_2_pool_2_max = tf.layers.max_pooling2d(parallel_2_2, 3, 2, 'same')
            #     # parallel_2_1_pool_2_max = tf.layers.max_pooling2d(parallel_2_1, 3, 2, 'same')
            #     # parallel_2_1_pool_4_max = tf.layers.max_pooling2d(parallel_2_1, 5, 4, 'same')
            #     # # concat pool
            #     # parallel_2_3_temp = tf.concat([parallel_2_3, parallel_2_2_pool_2_max, parallel_2_1_pool_4_max], -1)
            #     # parallel_2_3 = tf.layers.conv2d(parallel_2_3_temp, parallel_2_3.shape[-1], 1, 1, 'same', name='up/conv1/conv')
            #     # # concat pool
            #     # parallel_2_2_temp = tf.concat([parallel_2_2, parallel_2_1_pool_2_max], -1)
            #     # parallel_2_2 = tf.layers.conv2d(parallel_2_2_temp, parallel_2_2.shape[-1], 1, 1, 'same',name='up/conv2/conv')
            #
            #     # method 2 baochi xiangtong channel concat
            #     parallel_2_2_pool_2_max = tf.concat([tf.layers.max_pooling2d(parallel_2_2, 5, 2, 'same'),
            #                                          tf.layers.max_pooling2d(parallel_2_2, 7, 2, 'same')],-1)
            #
            #     parallel_2_1_pool_2_max = tf.concat([tf.layers.max_pooling2d(parallel_2_1, 5, 2, 'same'),
            #                                          tf.layers.max_pooling2d(parallel_2_1, 7, 2, 'same'),
            #                                          tf.layers.max_pooling2d(parallel_2_1, 9, 2, 'same'),
            #                                          tf.layers.max_pooling2d(parallel_2_1, 11, 2, 'same')],-1)
            #
            #     parallel_2_1_pool_4_max = tf.concat([tf.layers.max_pooling2d(parallel_2_1, 5, 4, 'same'),
            #                                          tf.layers.max_pooling2d(parallel_2_1, 7, 4, 'same'),
            #                                          tf.layers.max_pooling2d(parallel_2_1, 9, 4, 'same'),
            #                                          tf.layers.max_pooling2d(parallel_2_1, 11, 4, 'same')],-1)
            #     # concat pool
            #     parallel_2_3_temp = tf.concat([parallel_2_3, parallel_2_2_pool_2_max, parallel_2_1_pool_4_max], -1)
            #     parallel_2_3 = tf.layers.conv2d(parallel_2_3_temp, parallel_2_3.shape[-1], 1, 1, 'same', name='up/conv1/conv')
            #     # concat pool
            #     parallel_2_2_temp = tf.concat([parallel_2_2, parallel_2_1_pool_2_max], -1)
            #     parallel_2_2 = tf.layers.conv2d(parallel_2_2_temp, parallel_2_2.shape[-1], 1, 1, 'same',name='up/conv2/conv')
            #
            #     #
            #     up_2_1 = tf.layers.conv2d_transpose(inputs=down_4, filters=256, kernel_size=3,
            #                                    strides=2, padding='same', name='up/conv1')
            #     up_2_1 = tf.concat([up_2_1, parallel_2_3], axis=-1, name='up_concat_1' + str(1))
            #     is_2_5 = results_score(up_2_1, class_num[1], "instance_5")
            #     up_2_1_out = tf.image.resize_bilinear(is_2_5, [shape[1], shape[2]])
            #     self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_1_out, class_num[1], class_weights[1], name="redisual_layer_5/")
            #
            #     up_2_2 = tf.layers.conv2d_transpose(inputs=up_2_1, filters=128, kernel_size=3, strides=2, padding='same', name='up/conv2')
            #     up_2_2 = tf.concat([up_2_2, parallel_2_2], axis=-1,name='up_concat_2' + str(2))
            #     is_2_6 = results_score(up_2_2, class_num[1], "instance_6")
            #     up_2_2_out = tf.image.resize_bilinear(is_2_6, [shape[1], shape[2]])
            #     self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_2_out, class_num[1], class_weights[1], name="redisual_layer_6/")
            #
            #     up_2_3 = tf.layers.conv2d_transpose(inputs=up_2_2, filters=64, kernel_size=3, strides=2, padding='same', name='up/conv3')
            #     # maybe can cancel
            #     up_2_3 = tf.concat([up_2_3, parallel_2_1], axis=-1, name='up_concat_3' + str(3))
            #
            #     is_2_7 = results_score(up_2_3, class_num[1], "instance_7")
            #     up_2_3_out = tf.image.resize_bilinear(is_2_7, [shape[1], shape[2]])
            #     self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_3_out, class_num[1], class_weights[1], name="redisual_layer_7/")
            #
            #     up_2_4 = tf.layers.conv2d_transpose(up_2_3, filters=32, kernel_size=3, strides=2, padding='same', name='up/conv4')
            #     up_2_4_out = results_score(up_2_4, class_num[1], "instance_8")
            #     self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_4_out, class_num[1], class_weights[1], name="redisual_layer_8/")

            # parerllel down  all concat up conv
            # with tf.variable_scope("updown_layer_stage2"):
            #     #
            #     up_2_1 = tf.layers.conv2d_transpose(inputs=down_4, filters=256, kernel_size=3,
            #                                    strides=2, padding='same', name='up/conv1')
            #     #up_2_1 = tf.concat([up_2_1, parallel_2_3], axis=-1, name='up_concat_1' + str(1))
            #     down_3_concat = tf.concat([parallel_2_3, parallel_1_3, down_3], axis=-1, name='up_concat_1_pre' + str(1))
            #     down_3_concat = tf.layers.conv2d(down_3_concat, 256, 3, 1, 'same', name = 'up_concat_1/conv')
            #     up_2_1 = tf.concat([up_2_1, down_3_concat], axis=-1, name='up_concat_1' + str(1))
            #     # up_2_1 = tf.concat([up_2_1, parallel_2_3, parallel_1_3, down_3], axis=-1, name='up_concat_1' + str(1))
            #     is_2_5 = results_score(up_2_1, class_num[1], "instance_5")
            #     up_2_1_out = tf.image.resize_bilinear(is_2_5, [shape[1], shape[2]])
            #     self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_1_out, class_num[1], class_weights[1], name="redisual_layer_5/")
            #
            #     up_2_2 = tf.layers.conv2d_transpose(inputs=up_2_1, filters=128, kernel_size=3, strides=2, padding='same', name='up/conv2')
            #     #up_2_2 = tf.concat([up_2_2, parallel_2_2], axis=-1,name='up_concat_2' + str(2))
            #     down_2_concat = tf.concat([parallel_2_2, parallel_1_2, down_2], axis=-1, name='up_concat_2_pre' + str(1))
            #     down_2_concat = tf.layers.conv2d(down_2_concat, 128, 3, 1, 'same', name = 'up_concat_2/conv')
            #     up_2_2 = tf.concat([up_2_2, down_2_concat], axis=-1, name='up_concat_2' + str(2))
            #     #up_2_2 = tf.concat([up_2_2, parallel_2_2, parallel_1_2, down_2], axis=-1, name='up_concat_2' + str(2))
            #     is_2_6 = results_score(up_2_2, class_num[1], "instance_6")
            #     up_2_2_out = tf.image.resize_bilinear(is_2_6, [shape[1], shape[2]])
            #     self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_2_out, class_num[1], class_weights[1], name="redisual_layer_6/")
            #
            #     up_2_3 = tf.layers.conv2d_transpose(inputs=up_2_2, filters=64, kernel_size=3, strides=2, padding='same', name='up/conv3')
            #     # maybe can cancel
            #     #up_2_3 = tf.concat([up_2_3, parallel_2_1], axis=-1, name='up_concat_3' + str(3))
            #     down_1_concat = tf.concat([parallel_2_1, parallel_1_1, down_1], axis=-1, name='up_concat_3_pre' + str(1))
            #     down_1_concat = tf.layers.conv2d(down_1_concat, 64, 3, 1, 'same', name = 'up_concat_3/conv')
            #     up_2_3 = tf.concat([up_2_3, down_1_concat], axis=-1, name='up_concat_3' + str(3))
            #     #up_2_3 = tf.concat([up_2_3, parallel_2_1,parallel_1_1,down_1], axis=-1, name='up_concat_3' + str(3))
            #
            #     is_2_7 = results_score(up_2_3, class_num[1], "instance_7")
            #     up_2_3_out = tf.image.resize_bilinear(is_2_7, [shape[1], shape[2]])
            #     self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_3_out, class_num[1], class_weights[1], name="redisual_layer_7/")
            #
            #     up_2_4 = tf.layers.conv2d_transpose(up_2_3, filters=32, kernel_size=3, strides=2, padding='same', name='up/conv4')
            #     up_2_4_out = results_score(up_2_4, class_num[1], "instance_8")
            #     self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_4_out, class_num[1], class_weights[1], name="redisual_layer_8/")

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

            # parerllel up layer conv
            # with tf.variable_scope("updown_layer_stage2"):
            #     up_2_1 = tf.layers.conv2d_transpose(inputs=p_TD_1_3, filters=256, kernel_size=3,
            #                                    strides=2, padding='same', name='up/conv1')
            #     up_2_1 = tf.concat([up_2_1, parallel_2_3], axis=-1, name='up_concat_1' + str(1))
            #     # p1_1
            #     up_2_1_2 = tf.layers.conv2d_transpose(inputs=down_4, filters=256, kernel_size=3, strides=2, padding='same', name='up/conv_trans_1_2')
            #     up_2_1_2 = tf.concat([up_2_1, up_2_1_2], axis=-1)
            #     up_2_1_2 = tf.layers.conv2d(up_2_1_2, 256, 3, 1, 'same', name = 'up/conv1_p_1_1')
            #     #up_2_1_2 = res_block(up_2_1, 1, 64, 256, 3, train, name="redisual_layer_1")
            #     # p2_1
            #     #up_2_1_3 = tf.layers.conv2d_transpose(inputs=down_4, filters=256, kernel_size=3, strides=2, padding='same', name='up/conv_trans_1_3')
            #     up_2_1_3 = tf.concat([up_2_1_2], axis=-1)
            #     #up_2_1_3 = tf.concat([up_2_1_2, up_2_1_3], axis=-1)
            #     #up_2_1_3 = tf.concat([up_2_1_2, up_2_1_3,parallel_2_3], axis=-1)
            #     up_2_1_3 = tf.layers.conv2d(up_2_1_3, 256, 3, 1, 'same', name='up/conv1_p_2_1')
            #     #up_2_1_3 = res_block(up_2_1_2, 1, 64, 256, 3, train, name="redisual_layer_1_2")
            #
            #     is_2_5 = results_score(up_2_1_3, class_num[1], "instance_5")
            #     up_2_1_out = tf.image.resize_bilinear(is_2_5, [shape[1], shape[2]])
            #     self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_1_out, class_num[1], class_weights[1], name="redisual_layer_5/")
            #
            #     up_2_2 = tf.layers.conv2d_transpose(inputs=up_2_1, filters=128, kernel_size=3, strides=2, padding='same', name='up/conv2')
            #     up_2_2 = tf.concat([up_2_2, parallel_2_2], axis=-1,name='up_concat_2' + str(2))
            #     # p1_2
            #     up_2_2_2 = tf.layers.conv2d_transpose(inputs=up_2_1_2, filters=128, kernel_size=3, strides=2, padding='same', name='up/conv2_p_1_2')
            #     up_2_2_2 = tf.concat([up_2_2_2, up_2_2], axis=-1, name='up_concat_2' + str(2))
            #     up_2_2_2 = tf.layers.conv2d(up_2_2_2, 128, 3, 1, 'same', name='up/conv1_p_1_2')
            #     #up_2_2_2 = res_block(up_2_2_2, 1, 64, 128, 3, train, name="redisual_layer_2")
            #     # p2_2
            #     up_2_2_3 = tf.layers.conv2d_transpose(inputs=up_2_1_3, filters=128, kernel_size=3, strides=2, padding='same', name='up/conv2_p_2_2')
            #     up_2_2_3 = tf.concat([up_2_2_3, up_2_2_2], axis=-1, name='up_concat_2_2' + str(2))
            #     #up_2_2_3 = tf.concat([up_2_2_3, up_2_2_2,parallel_2_2], axis=-1, name='up_concat_2_2' + str(2))
            #     up_2_2_3 = tf.layers.conv2d(up_2_2_3, 128, 3, 1, 'same', name='up/conv1_p_2_2')
            #     #up_2_2_3 = res_block(up_2_2_3, 1, 64, 128, 3, train, name="redisual_layer_2_2")
            #
            #     is_2_6 = results_score(up_2_2_3, class_num[1], "instance_6")
            #     up_2_2_out = tf.image.resize_bilinear(is_2_6, [shape[1], shape[2]])
            #     self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_2_out, class_num[1], class_weights[1], name="redisual_layer_6/")
            #
            #     up_2_3 = tf.layers.conv2d_transpose(inputs=up_2_2, filters=64, kernel_size=3, strides=2, padding='same', name='up/conv3')
            #     up_2_3 = tf.concat([up_2_3, parallel_2_1], axis=-1, name='up_concat_3' + str(3))
            #
            #     #p1_3
            #     up_2_3_2 = tf.layers.conv2d_transpose(inputs=up_2_2_2, filters=64, kernel_size=3, strides=2, padding='same', name='up/conv2_p_1_3')
            #     up_2_3_2 = tf.concat([up_2_3_2, up_2_3], axis=-1)
            #     up_2_3_2 = tf.layers.conv2d(up_2_3_2, 64, 3, 1, 'same', name='up/conv1_p_1_3')
            #     #up_2_3_2 = res_block(up_2_3_2, 1, 64, 64, 3, train, name="redisual_layer_3")
            #     #p2_3
            #     up_2_3_3 = tf.layers.conv2d_transpose(inputs=up_2_2_3, filters=64, kernel_size=3, strides=2, padding='same', name='up/conv2_p_2_3')
            #     up_2_3_3 = tf.concat([up_2_3_3, up_2_3_2], axis=-1)
            #     #up_2_3_3 = tf.concat([up_2_3_3, up_2_3_2, parallel_2_1], axis=-1)
            #     up_2_3_3 = tf.layers.conv2d(up_2_3_3, 64, 3, 1, 'same', name='up/conv1_p_2_3')
            #     #up_2_3_3 = res_block(up_2_3_3, 1, 64, 64, 3, train, name="redisual_layer_3_2")
            #
            #     is_2_7 = results_score(up_2_3_3, class_num[1], "instance_7")
            #     up_2_3_out = tf.image.resize_bilinear(is_2_7, [shape[1], shape[2]])
            #     self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_3_out, class_num[1], class_weights[1], name="redisual_layer_7/")
            #
            #     #up_2_4 = tf.layers.conv2d_transpose(up_2_3, filters=32, kernel_size=3, strides=2, padding='same', name='up/conv4')
            #     # p1_4
            #     up_2_3_3 = tf.concat([up_2_3_3, up_2_3_2, up_2_3], axis=-1)
            #     up_2_4_2 = tf.layers.conv2d_transpose(inputs=up_2_3_3, filters=32, kernel_size=3, strides=2, padding='same', name='up/conv2_p_1_4')
            #     #up_2_4_2 = tf.concat([up_2_4_2, up_2_4], axis=-1)
            #     #up_2_4_2 = tf.layers.conv2d(up_2_4_2, 32, 3, 1, 'same', name='up/conv1_p_1_4')
            #     #up_2_4_2 = res_block(up_2_4_2, 1, 32, 32, 3, train, name="redisual_layer_4")
            #
            #     up_2_4_out = results_score(up_2_4_2, class_num[1], "instance_8")
            #     self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_4_out, class_num[1], class_weights[1], name="redisual_layer_8/")

            # out = up4_out
            with tf.variable_scope("instance_layer"):
                #out_2 = tf.concat([out4, up_2_1_out, up_2_2_out, up_2_3_out, up_2_4_out], -1)
                out_2 = tf.concat([out4, up_2_1_out, up_2_2_out, up_2_3_out, up_2_4_out], -1)
                #out = tf.concat([up4_out, up3_out, up2_out, up1_out, out4, out3, out2, out1], -1)
                out_2 = results_score(out_2, class_num[1], "instance_out_layer4_2")
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, out_2, class_num[1], class_weights[1], name="out_2/")
            return tf.nn.softmax(out_2)

    def build_multi_parallel_pyramid_attention_net(self, inputs_org, class_num, class_weights, train=True):
        with tf.variable_scope("stage2"):
            inputs = self.init_img(inputs_org, train)
            shape = tf.shape(inputs_org)
            self.mid_result = []
            with tf.variable_scope("layer1"):

                # inputs = concatenate_smaller_images(inputs, inputs_org)
                # # crop
                # inputs = input_crop_layer1(inputs_org, inputs, "X", train)

                down_1 = redisual_block(inputs, 3, 64, 3, 0, train, 1, "redisual_layer_1")
                #attention
                #down_1 = atblock(down_1, 64, scope='at_block_down_1')
                TD_1 = transition_down(down_1, down_1.get_shape()[-1], 3, 0, train, 2, 'transition_down_1')
                # parallel
                parallel_1_1 = redisual_block(down_1, 3, 64, 3, 0, train, 1, "parallel_redisual_layer_1_1")
                # attention
                #parallel_1_1 = atblock(down_1, 64, scope='at_block_parallel_1_1')
                #parallel_1_1 = atblock(parallel_1_1, 64, scope='at_block_parallel_1_1')
                p_TD_1_1 = transition_down(parallel_1_1, parallel_1_1.get_shape()[-1], 3, 0, train, 2, 'parallel_transition_down_1_1')

                parallel_2_1 = redisual_block(parallel_1_1, 3, 64, 3, 0, train, 1, "parallel_redisual_layer_2_1")
                # attention
                #parallel_2_1 = atblock(parallel_1_1, 64, scope='at_block_parallel_2_1')
                #parallel_2_1 = atblock(parallel_2_1, 64, scope='at_block_parallel_2_1')
                p_TD_2_1 = transition_down(parallel_2_1, parallel_2_1.get_shape()[-1], 3, 0, train, 2,
                                           'parallel_transition_down_2_1')

                # parallel_3_1 = redisual_block(parallel_2_1, 3, 64, 3, 0, train, 1, "parallel_redisual_layer_3_1")
                # p_TD_3_1 = transition_down(parallel_3_1, parallel_3_1.get_shape()[-1], 3, 0, train, 2,
                #                            'parallel_transition_down_3_1')

                #self.mid_result.append(TD_1)

                #TD_1 = results_score(TD_1, 32, "pre_instance_1")
                # is1 = tf.image.resize_bilinear(TD_1, [shape[1], shape[2]])
                # out1 = results_score(is1, class_num, "instance_1")

                # is1_2 = results_score(TD_1, class_num[1], "instance_1_2")
                # out1_2 = tf.image.resize_bilinear(is1_2, [shape[1], shape[2]])
                # self.loss_stage2 += dynamic_sparse_weighted_cross_entropy(self.output_2, out1_2, class_num[1], class_weights[1],
                #                                                    name="redisual_layer_1_2/")


            with tf.variable_scope("layer2"):

                # crop
                # TD_1 = input_crop_layer2(inputs_org, TD_1, "X", train)
                # p_TD_1_1 = input_crop_layer2(inputs_org, p_TD_1_1, "X", train)
                # p_TD_2_1 = input_crop_layer2(inputs_org, p_TD_2_1, "X", train)

                down_2 = redisual_block(TD_1, 3, 128, 3, 0, train, 1, "redisual_layer_2")
                #attention
                #down_2 = atblock(down_2, 128, scope='at_block_down_2')
                TD_2 = transition_down(down_2, down_2.get_shape()[-1], 3, 0, train, 2, 'transition_down_2')
                # parallel
                p_TD_1_1 = tf.concat([down_2, p_TD_1_1], axis=-1)
                parallel_1_2 = redisual_block(p_TD_1_1, 3, 128, 3, 0, train, 1, "parallel_redisual_layer_1_2")
                # attention
                #parallel_1_2 = atblock(parallel_1_2, 128, scope='at_block_parallel_1_2')
                p_TD_1_2 = transition_down(parallel_1_2, parallel_1_2.get_shape()[-1], 3, 0, train, 2, 'parallel_transition_down_1_2')

                p_TD_2_1 = tf.concat([parallel_1_2, p_TD_2_1], axis=-1)
                parallel_2_2 = redisual_block(p_TD_2_1, 3, 128, 3, 0, train, 1, "parallel_redisual_layer_2_2")
                # attention
                #parallel_2_2 = atblock(parallel_2_2, 128, scope='at_block_parallel_2_2')
                p_TD_2_2 = transition_down(parallel_2_2, parallel_2_2.get_shape()[-1], 3, 0, train, 2,
                                           'parallel_transition_down_2_2')

                # p_TD_3_1 = tf.concat([parallel_2_2, p_TD_3_1], axis=-1)
                # parallel_3_2 = redisual_block(p_TD_3_1, 3, 128, 3, 0, train, 1, "parallel_redisual_layer_3_2")
                # p_TD_3_2 = transition_down(parallel_3_2, parallel_3_2.get_shape()[-1], 3, 0, train, 2,
                #                            'parallel_transition_down_3_2')

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
                #attention
                #down_3 = atblock(down_3, 256, scope='at_block_down_3')
                TD_3 = transition_down(down_3, down_3.get_shape()[-1], 3, 0, train, 2, 'transition_down_3')

                # parallel
                p_TD_1_2 = tf.concat([down_3, p_TD_1_2], axis=-1)
                parallel_1_3 = redisual_block(p_TD_1_2, 3, 256, 3, 0, train, 1, "parallel_redisual_layer_1_3")
                # attention
                #parallel_1_3 = atblock(parallel_1_3, 256, scope='at_block_parallel_1_3')
                p_TD_1_3 = transition_down(parallel_1_3, parallel_1_3.get_shape()[-1], 3, 0, train, 2, 'parallel_transition_down_1_3')

                p_TD_2_2 = tf.concat([parallel_1_3, p_TD_2_2], axis=-1)
                parallel_2_3 = redisual_block(p_TD_2_2, 3, 256, 3, 0, train, 1, "parallel_redisual_layer_2_3")
                # attention
                #parallel_2_3 = atblock(parallel_2_3, 256, scope='at_block_parallel_2_3')
                p_TD_2_3 = transition_down(parallel_2_3, parallel_2_3.get_shape()[-1], 3, 0, train, 2, 'parallel_transition_down_2_3')

                # p_TD_3_2 = tf.concat([parallel_2_3, p_TD_3_2], axis=-1)
                # parallel_3_3 = redisual_block(p_TD_3_2, 3, 256, 3, 0, train, 1, "parallel_redisual_layer_3_3")
                # p_TD_3_3 = transition_down(parallel_3_3, parallel_3_3.get_shape()[-1], 3, 0, train, 2, 'parallel_transition_down_3_3')

                #self.mid_result.append(TD_3)
                #TD_3 = results_score(TD_3, 32, "pre_instance_3")
                # is3 = tf.image.resize_bilinear(TD_3, [shape[1], shape[2]])
                # out3 = results_score(is3, class_num, "instance_3")

                # is3_2 = results_score(TD_3, class_num[1], "instance_3_2")
                # out3_2 = tf.image.resize_bilinear(is3_2, [shape[1], shape[2]])
                # self.loss_stage2 += dynamic_sparse_weighted_cross_entropy(self.output_2, out3_2, class_num[1], class_weights[1],
                #                                                    name="redisual_layer_3_2/")

            with tf.variable_scope("layer4"):

                #down_4 = redisual_block(p_TD_2_3, 3, 256, 3, 0, train, 1, "redisual_layer_4")

                down_4 = tf.concat([TD_3, p_TD_1_3, p_TD_2_3], axis=-1)
                # down_4 = tf.concat([TD_3, p_TD_1_3, p_TD_2_3, p_TD_3_3], axis=-1)
                down_4 = redisual_block(down_4, 3, 256, 3, 0, train, 1, "redisual_layer_4")
                # attention
                #down_4 = atblock(down_4, 256, scope='at_block_down_4')

                #TD_4 = down_4
                #self.mid_result.append(down_4)
                #TD_4 = results_score(TD_4, 32, "pre_instance_4")
                # is4 = tf.image.resize_bilinear(TD_4, [shape[1], shape[2]])
                # out4 = results_score(is4, class_num, "instance_4")
                # is4 = results_score(down_4, class_num[1], "instance_4")
                # out4 = tf.image.resize_bilinear(is4, [shape[1], shape[2]])
                # self.loss += dynamic_sparse_weighted_cross_entropy(self.output, out4, class_num[1], class_weights[1],
                #                                                           name="redisual_layer_4/")

                is4 = results_score(down_4, class_num[1], "instance_4")
                out4 = tf.image.resize_bilinear(is4, [shape[1], shape[2]])
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, out4, class_num[1], class_weights[1],
                                                                          name="redisual_layer_4/")
            # U up conv
            with tf.variable_scope("updown_layer_stage2"):
                up_2_1 = tf.layers.conv2d_transpose(inputs=down_4, filters=256, kernel_size=3,
                                               strides=2, padding='same', name='up/conv1')
                up_2_1 = tf.concat([up_2_1, parallel_2_3], axis=-1, name='up_concat_1' + str(1))
                is_2_5 = results_score(up_2_1, class_num[1], "instance_5")
                up_2_1_out = tf.image.resize_bilinear(is_2_5, [shape[1], shape[2]])
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_1_out, class_num[1], class_weights[1], name="redisual_layer_5/")

                up_2_2 = tf.layers.conv2d_transpose(inputs=up_2_1, filters=128, kernel_size=3, strides=2, padding='same', name='up/conv2')
                up_2_2 = tf.concat([up_2_2, parallel_2_2], axis=-1,name='up_concat_2' + str(2))
                is_2_6 = results_score(up_2_2, class_num[1], "instance_6")
                up_2_2_out = tf.image.resize_bilinear(is_2_6, [shape[1], shape[2]])
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_2_out, class_num[1], class_weights[1], name="redisual_layer_6/")

                up_2_3 = tf.layers.conv2d_transpose(inputs=up_2_2, filters=64, kernel_size=3, strides=2, padding='same', name='up/conv3')
                up_2_3 = tf.concat([up_2_3, parallel_2_1], axis=-1, name='up_concat_3' + str(3))
                is_2_7 = results_score(up_2_3, class_num[1], "instance_7")
                up_2_3_out = tf.image.resize_bilinear(is_2_7, [shape[1], shape[2]])
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_3_out, class_num[1], class_weights[1], name="redisual_layer_7/")

                up_2_4 = tf.layers.conv2d_transpose(up_2_3, filters=32, kernel_size=3, strides=2, padding='same', name='up/conv4')
                up_2_4_out = results_score(up_2_4, class_num[1], "instance_8")
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_4_out, class_num[1], class_weights[1], name="redisual_layer_8/")

            # out = up4_out
            with tf.variable_scope("instance_layer"):
                #out_2 = tf.concat([out4, up_2_1_out, up_2_2_out, up_2_3_out, up_2_4_out], -1)
                out_2 = tf.concat([out4, up_2_1_out, up_2_2_out, up_2_3_out, up_2_4_out], -1)
                #out = tf.concat([up4_out, up3_out, up2_out, up1_out, out4, out3, out2, out1], -1)
                out_2 = results_score(out_2, class_num[1], "instance_out_layer4_2")
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, out_2, class_num[1], class_weights[1], name="out_2/")
            return tf.nn.softmax(out_2)

    def build_cross_parallel_pyramid_net(self, inputs_org, class_num, class_weights, train=True):
        with tf.variable_scope("stage2"):
            inputs = self.init_img(inputs_org, train)
            shape = tf.shape(inputs_org)
            self.mid_result = []
            with tf.variable_scope("layer1"):
                down_1 = redisual_block(inputs, 3, 64, 3, 0, train, 1, "redisual_layer_1")
                TD_1 = transition_down(down_1, down_1.get_shape()[-1], 3, 0, train, 2, 'transition_down_1')

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

                # parallel
                cross_parallel_input = tf.concat([down_2, TD_1], axis=-1)
                parallel_1 = redisual_block(cross_parallel_input, 3, 128, 3, 0, train, 1, "cross_parallel_redisual_layer_1")
                p_TD_1 = transition_down(parallel_1, parallel_1.get_shape()[-1], 3, 0, train, 2, 'cross_parallel_transition_down_1')

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

                # parallel
                parallel_2 = tf.concat([down_3, TD_2, p_TD_1], axis=-1)
                parallel_2 = redisual_block(parallel_2, 3, 256, 3, 0, train, 1, "cross_parallel_redisual_layer_2")
                p_TD_2 = transition_down(parallel_2, parallel_2.get_shape()[-1], 3, 0, train, 2,
                                         'cross_parallel_transition_down_2')
                #self.mid_result.append(TD_3)
                #TD_3 = results_score(TD_3, 32, "pre_instance_3")
                # is3 = tf.image.resize_bilinear(TD_3, [shape[1], shape[2]])
                # out3 = results_score(is3, class_num, "instance_3")

                # is3_2 = results_score(TD_3, class_num[1], "instance_3_2")
                # out3_2 = tf.image.resize_bilinear(is3_2, [shape[1], shape[2]])
                # self.loss_stage2 += dynamic_sparse_weighted_cross_entropy(self.output_2, out3_2, class_num[1], class_weights[1],
                #                                                    name="redisual_layer_3_2/")

            with tf.variable_scope("layer4"):
                down_4_temp = tf.concat([TD_3, p_TD_2], axis=-1)
                down_4 = redisual_block(down_4_temp, 3, 256, 3, 0, train, 1, "redisual_layer_4")

                #TD_4 = down_4
                #self.mid_result.append(down_4)
                #TD_4 = results_score(TD_4, 32, "pre_instance_4")
                # is4 = tf.image.resize_bilinear(TD_4, [shape[1], shape[2]])
                # out4 = results_score(is4, class_num, "instance_4")
                # is4 = results_score(down_4, class_num[1], "instance_4")
                # out4 = tf.image.resize_bilinear(is4, [shape[1], shape[2]])
                # self.loss += dynamic_sparse_weighted_cross_entropy(self.output, out4, class_num[1], class_weights[1],
                #                                                           name="redisual_layer_4/")

                is4 = results_score(down_4, class_num[1], "instance_4")
                out4 = tf.image.resize_bilinear(is4, [shape[1], shape[2]])
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, out4, class_num[1], class_weights[1],
                                                                          name="redisual_layer_4/")

            with tf.variable_scope("updown_layer_stage2"):
                up_2_1 = tf.layers.conv2d_transpose(inputs=down_4, filters=256, kernel_size=3,
                                               strides=2, padding='same', name='up/conv1')
                up_2_1 = tf.concat([up_2_1, parallel_2], axis=-1, name='up_concat_1' + str(1))
                is_2_5 = results_score(up_2_1, class_num[1], "instance_5")
                up_2_1_out = tf.image.resize_bilinear(is_2_5, [shape[1], shape[2]])
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_1_out, class_num[1], class_weights[1], name="redisual_layer_5/")

                up_2_2 = tf.layers.conv2d_transpose(inputs=up_2_1, filters=128, kernel_size=3, strides=2, padding='same', name='up/conv2')
                up_2_2 = tf.concat([up_2_2, parallel_1], axis=-1,name='up_concat_2' + str(2))
                is_2_6 = results_score(up_2_2, class_num[1], "instance_6")
                up_2_2_out = tf.image.resize_bilinear(is_2_6, [shape[1], shape[2]])
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_2_out, class_num[1], class_weights[1], name="redisual_layer_6/")

                up_2_3 = tf.layers.conv2d_transpose(inputs=up_2_2, filters=64, kernel_size=3, strides=2, padding='same', name='up/conv3')
                up_2_3 = tf.concat([up_2_3, down_1], axis=-1, name='up_concat_3' + str(3))
                is_2_7 = results_score(up_2_3, class_num[1], "instance_7")
                up_2_3_out = tf.image.resize_bilinear(is_2_7, [shape[1], shape[2]])
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_3_out, class_num[1], class_weights[1], name="redisual_layer_7/")

                up_2_4 = tf.layers.conv2d_transpose(up_2_3, filters=32, kernel_size=3, strides=2, padding='same', name='up/conv4')
                up_2_4_out = results_score(up_2_4, class_num[1], "instance_8")
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_4_out, class_num[1], class_weights[1], name="redisual_layer_8/")

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
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, out_2, class_num[1], class_weights[1], name="out_2/")
            return tf.nn.softmax(out_2)

    def build_net_new(self, inputs_org, class_num, class_weights, train=True):

        with tf.variable_scope("stage1"):
            shape = tf.shape(inputs_org)
            inputs_init = init_conv(inputs_org, 64, 7, train)
            # 60 * 60
            pool_block = res_transition_down(inputs_init, 64, 256, kernel_size=3, strides=1,
                        is_dropout=False, dropout=0.2, train=True)

            with tf.variable_scope("layer1"):

                down_1 = res_block(pool_block, 2, 64, 128, 3, train, name="redisual_layer_1")
                TD_1 = res_transition_down(down_1, 128, 512, kernel_size=3, strides=2,
                                                 is_dropout=False, dropout=0.2, train=True)

                #self.mid_result.append(TD_1)

                #TD_1 = results_score(TD_1, 32, "pre_instance_1")
                # is1 = tf.image.resize_bilinear(TD_1, [shape[1], shape[2]])
                # out1 = results_score(is1, class_num, "instance_1")

                # is1_2 = results_score(TD_1, class_num[1], "instance_1_2")
                # out1_2 = tf.image.resize_bilinear(is1_2, [shape[1], shape[2]])
                # self.loss_stage2 += dynamic_sparse_weighted_cross_entropy(self.output_2, out1_2, class_num[1], class_weights[1],
                #                                                    name="redisual_layer_1_2/")


            with tf.variable_scope("layer2"):

                down_2 = res_block(TD_1, 3, 128, 512, 3, train, name="redisual_layer_2")
                TD_2 = res_transition_down(down_2, 256, 1024, kernel_size=3, strides=2,
                                                 is_dropout=False, dropout=0.2, train=True)
                #self.mid_result.append(TD_2)
                #TD_2 = results_score(TD_2, 32, "pre_instance_2")
                # is2 =  tf.image.resize_bilinear(TD_2, [shape[1], shape[2]])
                # out2 = results_score(is2, class_num, "instance_2")

                # is2_2 = results_score(TD_2, class_num[1], "instance_2_2")
                # out2_2 = tf.image.resize_bilinear(is2_2, [shape[1], shape[2]])
                # self.loss_stage2 += dynamic_sparse_weighted_cross_entropy(self.output_2, out2_2, class_num[1], class_weights[1],
                #                                                    name="redisual_layer_2_2/")

            with tf.variable_scope("layer3"):

                down_3 = res_block(TD_2, 5, 256, 1024, 3, train, name="redisual_layer_3")
                TD_3 = res_transition_down(down_3, 512, 2048, kernel_size=3, strides=2,
                                                 is_dropout=False, dropout=0.2, train=True)

                #self.mid_result.append(TD_3)
                #TD_3 = results_score(TD_3, 32, "pre_instance_3")
                # is3 = tf.image.resize_bilinear(TD_3, [shape[1], shape[2]])
                # out3 = results_score(is3, class_num, "instance_3")

                # is3_2 = results_score(TD_3, class_num[1], "instance_3_2")
                # out3_2 = tf.image.resize_bilinear(is3_2, [shape[1], shape[2]])
                # self.loss_stage2 += dynamic_sparse_weighted_cross_entropy(self.output_2, out3_2, class_num[1], class_weights[1],
                #                                                    name="redisual_layer_3_2/")

            with tf.variable_scope("layer4"):

                down_4 = res_block(TD_3, 3, 512, 2048, 3, train, name="redisual_layer_4")
                #TD_4 = down_4
                #self.mid_result.append(down_4)
                #TD_4 = results_score(TD_4, 32, "pre_instance_4")
                # is4 = tf.image.resize_bilinear(TD_4, [shape[1], shape[2]])
                # out4 = results_score(is4, class_num, "instance_4")
                is4 = results_score(down_4, class_num[1], "instance_4")
                out4 = tf.image.resize_bilinear(is4, [shape[1], shape[2]])
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, out4, class_num[1], class_weights[1],
                                                                          name="redisual_layer_4/")

            with tf.variable_scope("updown_layer_stage"):

                # small
                # up_2_1 = tf.layers.conv2d_transpose(inputs=down_4, filters=256, kernel_size=3,
                #                                strides=2, padding='same', name='up/conv1')
                # down_3_shape = tf.shape(down_3)
                # up_2_1 = tf.image.resize_bilinear(up_2_1, [down_3_shape[1], down_3_shape[2]])
                # up_2_1 = tf.concat([up_2_1, down_3], axis=-1, name='up_concat_1' + str(1))
                # is_2_5 = results_score(up_2_1, class_num[1], "instance_5")
                # up_2_1_out = tf.image.resize_bilinear(is_2_5, [shape[1], shape[2]])
                # self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_1_out, class_num[1], class_weights[1], name="redisual_layer_5/")
                #
                # up_2_2 = tf.layers.conv2d_transpose(inputs=up_2_1, filters=128, kernel_size=3, strides=2, padding='same', name='up/conv2')
                # up_2_2 = tf.concat([up_2_2, down_2], axis=-1,name='up_concat_2' + str(2))
                # is_2_6 = results_score(up_2_2, class_num[1], "instance_6")
                # up_2_2_out = tf.image.resize_bilinear(is_2_6, [shape[1], shape[2]])
                # self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_2_out, class_num[1], class_weights[1], name="redisual_layer_6/")
                #
                # up_2_3 = tf.layers.conv2d_transpose(inputs=up_2_2, filters=64, kernel_size=3, strides=2, padding='same', name='up/conv3')
                # up_2_3 = tf.concat([up_2_3, down_1], axis=-1, name='up_concat_3' + str(3))
                # is_2_7 = results_score(up_2_3, class_num[1], "instance_7")
                # up_2_3_out = tf.image.resize_bilinear(is_2_7, [shape[1], shape[2]])
                # self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_3_out, class_num[1], class_weights[1], name="redisual_layer_7/")
                #
                # up_2_4 = tf.layers.conv2d_transpose(up_2_3, filters=32, kernel_size=3, strides=4, padding='same', name='up/conv4')
                # up_2_4_out = results_score(up_2_4, class_num[1], "instance_8")
                # self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_4_out, class_num[1], class_weights[1], name="redisual_layer_8/")

                # new
                up_2_1 = tf.layers.conv2d_transpose(inputs=down_4, filters=1024, kernel_size=3,
                                               strides=2, padding='same', name='up/conv1')
                down_3_shape = tf.shape(down_3)
                up_2_1 = tf.image.resize_bilinear(up_2_1, [down_3_shape[1], down_3_shape[2]])
                up_2_1 = tf.concat([up_2_1, down_3], axis=-1, name='up_concat_1' + str(1))
                is_2_5 = results_score(up_2_1, class_num[1], "instance_5")
                up_2_1_out = tf.image.resize_bilinear(is_2_5, [shape[1], shape[2]])
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_1_out, class_num[1], class_weights[1], name="redisual_layer_5/")

                up_2_2 = tf.layers.conv2d_transpose(inputs=up_2_1, filters=512, kernel_size=3, strides=2, padding='same', name='up/conv2')
                up_2_2 = tf.concat([up_2_2, down_2], axis=-1,name='up_concat_2' + str(2))
                is_2_6 = results_score(up_2_2, class_num[1], "instance_6")
                up_2_2_out = tf.image.resize_bilinear(is_2_6, [shape[1], shape[2]])
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_2_out, class_num[1], class_weights[1], name="redisual_layer_6/")

                up_2_3 = tf.layers.conv2d_transpose(inputs=up_2_2, filters=256, kernel_size=3, strides=2, padding='same', name='up/conv3')
                up_2_3 = tf.concat([up_2_3, down_1], axis=-1, name='up_concat_3' + str(3))
                is_2_7 = results_score(up_2_3, class_num[1], "instance_7")
                up_2_3_out = tf.image.resize_bilinear(is_2_7, [shape[1], shape[2]])
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_3_out, class_num[1], class_weights[1], name="redisual_layer_7/")

                up_2_4 = tf.layers.conv2d_transpose(up_2_3, filters=128, kernel_size=3, strides=4, padding='same', name='up/conv4')
                up_2_4_out = results_score(up_2_4, class_num[1], "instance_8")
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, up_2_4_out, class_num[1], class_weights[1], name="redisual_layer_8/")

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
                self.loss += dynamic_sparse_weighted_cross_entropy(self.output, out_2, class_num[1], class_weights[1], name="out_2/")
            return tf.nn.softmax(out_2)

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

