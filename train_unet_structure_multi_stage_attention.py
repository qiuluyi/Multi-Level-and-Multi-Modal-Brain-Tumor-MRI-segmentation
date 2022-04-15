import logging
import sys
import tensorflow as tf
import scipy as scp
import os
import numpy as np
import cv2 as cv


# from MHA_change.readMHATesting import BRATS2015Test as Brats
from Fuzzy.readMHA_something import BRATS2015 as Brats
from Fuzzy.model_unet_structure_multi_stage_attention import model
from MHA_change.Region import Region
from MHA_change.utils import utils
from Fuzzy.config import cfg
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

# Parse Options
path = 'RESNEXT_unet_20200110_04c_13' + cfg.dimensional + '/'
saveImgPath = 'image_result/Image_train_' + path
saveTestImgPath = 'image_result/Image_Test_' + path
modelSavePath = 'models/dilate_model_' + path
modelRestorePath = 'models/dilate_model_' + path + str(cfg.restore_epoch) +'epoch/'
log_path = 'logs/train_' + path

if not os.path.exists(saveImgPath):
    os.makedirs(saveImgPath)
if not os.path.exists(saveTestImgPath):
    os.makedirs(saveTestImgPath)

def save_to():
    if not os.path.exists(cfg.data_results):
        os.mkdir(cfg.data_results)
    if cfg.is_training:
        data_results = cfg.data_results + '/' + cfg.dimensional + '_data_results.csv'
        if os.path.exists(data_results):
            os.remove(data_results)
        fd_train_data = open(data_results, 'w')
        fd_train_data.write('global_step, loss_1, loss_2, acc_1, acc_2, dice, dice2, dice3, all_loss_show\n')
        return(fd_train_data)

ra = np.random.randint(0, 274) # ra = np.random.randint(0, 274)  # random to choose the begining brats
brats = Brats(8,0,train = cfg.is_training,test=cfg.is_testing, both_hgg_lgg=cfg.both_hgg_lgg, validation_set_num=cfg.validation_set_num)
# load_npy = load_from_npy()

def lets_go():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.8
    with tf.Session(config=config) as sess:
    #with tf.Session() as sess:

        # from tensorflow.python import debug as tf_debug
        # # # 使用tf_debug的wrapper来包裹原来的session，使得启动后能进入
        # # #   CLI调试界面.
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # # # 添加过滤器，用于监测 NAN 和 INF
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        class_weight = [[1,5],[1, 5, 3, 4, 10]]
        class_num = [2,5]
        images = tf.placeholder("float32", [None, 240, 240, 4])
        outputs = tf.placeholder("int32", [None, 240, 240  ])
        net = model(class_num=class_num, img=images, out=outputs, class_weights=class_weight)  # ,train=train_able
        print('build network success')

        all_loss = tf.get_collection("loss")
        init = tf.global_variables_initializer()
        sess.run(init)
        # tf.image.resize_images()
        saver = tf.train.Saver()
        # merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(log_path, sess.graph)
        tf.global_variables_initializer().run()
        ckpt = tf.train.get_checkpoint_state(modelRestorePath)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restore")

        # saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=2)
        if cfg.is_training:
            # save result data to csv
            fd_train_data = save_to()
            for epoch in range(cfg.epoch - cfg.restore_epoch):
                print("Training for epoch %d/%d:" % (epoch + cfg.restore_epoch + 1, cfg.epoch))
                for i in tqdm(range((brats.train_batch * 155) // cfg.batch_size), total=((brats.train_batch * 155) // cfg.batch_size),
                              ncols=70, leave=False, unit='b'):
                    # for i in range(total_train_batch//cfg.batch_size):
                    img_org, label = brats.next_train_batch(cfg.batch_size)

                    feed_dict = {images: img_org, outputs: label}
                    _ = sess.run(net.train_op, feed_dict=feed_dict)
                    if i % 100 == 0 and np.max(label) >= 1:
                        loss,  result, global_step, label_min, acc, all_loss_show = sess.run(
                            [#merged,
                             net.loss,
                             net.result,
                             net.global_step,
                             net.output,
                             net.acc,
                             all_loss
                             ],
                            feed_dict=feed_dict)

                        #train_writer.add_summary(summary, global_step_2)
                        dataR1, labelR1 = Region.Region1(np.argmax(result, axis=-1), label_min)
                        dice = utils.dice_coef(dataR1, labelR1)
                        dataR2, labelR2 = Region.Region2(np.argmax(result, axis=-1), label_min)
                        dice2 = utils.dice_coef(dataR2, labelR2)
                        dataR3, labelR3 = Region.Region3(np.argmax(result, axis=-1), label_min)
                        dice3 = utils.dice_coef(dataR3, labelR3)
                        # print Image()

                        img_multimodal = img_org[0,:,:,0]
                        for modal in range(len(img_org[0,0,0,:])-1):
                            img_multimodal = np.concatenate((img_multimodal, img_org[0,:,:,modal+1]),axis=0)
                        input_save = img_org[0]
                        out_save = np.concatenate((label_min[0, :, :], np.argmax(result[0], axis=-1)), axis=1)
                        out_save_color = utils.color_image(out_save)
                        scp.misc.imsave(saveImgPath + brats.mhaName + str(i) + 'multimodal_img.png', img_multimodal)
                        scp.misc.imsave(saveImgPath + brats.mhaName + str(i) + 'input_img_2.png', input_save)
                        scp.misc.imsave(saveImgPath + brats.mhaName + str(i) + 'net_out_2.png', out_save_color)

                        # print(global_step, loss, acc, dice, dice2, dice3, all_loss_show)
                        fd_train_data.write(str(global_step) + ',' + str(loss) + ','
                                            + str(acc) + ','+ str(dice) + ',' + str(dice2) + ',' + str(dice3) + ',' + str(
                            all_loss_show) + '\n')
                        fd_train_data.flush()
                        # saver.save(sess, modelSavePath , global_step=net.global_step)
                if (epoch + cfg.restore_epoch + 1) % 5 == 0:
                    savepath = modelSavePath + str(epoch + cfg.restore_epoch + 1) + "epoch/"
                    if not os.path.exists(savepath):
                        os.makedirs(savepath)
                    saver.save(sess, savepath, global_step= epoch + cfg.restore_epoch + 1)
                elif (epoch + cfg.restore_epoch + 1) == 35:
                    savepath = modelSavePath + str(epoch + cfg.restore_epoch + 1) + "epoch/"
                    if not os.path.exists(savepath):
                        os.makedirs(savepath)
                    saver.save(sess, savepath, global_step= epoch + cfg.restore_epoch + 1)
            # close file stream
            fd_train_data.close()
        elif not cfg.is_testing:
            # for i in range(16):
            #     img, label = brats.next_train_batch(cfg.batch_size)
            #     feed_dict = {images: img, outputs: label}
            #     mid_result = sess.run(net.mid_result, feed_dict=feed_dict)
            #     j = 0
            #     for resultsss in mid_result:
            #         j += 1
            #         np.save("mid_result/data/" + str(i) + "_" + str(j) + ".npy", resultsss)
            for i in tqdm(range(((155) // cfg.batch_size + 1) * brats.test_batch), total=((155 // cfg.batch_size + 1) * brats.test_batch),
                          ncols=70, leave=False, unit='b'):
                img_org, label_2 = brats.next_test_batch(cfg.batch_size)
                feed_dict = {images: img_org}
                result = sess.run(net.result, feed_dict=feed_dict)
                result_color = utils.color_image(np.argmax(result[0], axis=2))
                if i % 50 == 0:
                    scp.misc.imsave(saveTestImgPath + brats.mhaName + str(i) + 'fcn_img.png', img_org[0])
                    # scp.misc.imsave(saveTestImgPath + str(i) + 'fcn_label.png', label_color)
                    scp.misc.imsave(saveTestImgPath + brats.mhaName + str(i) + 'fcn_result.png', result_color)
                brats.saveItk(np.argmax(result, axis=3), "multi_stage")
        else:
            for i in tqdm(range((155 // cfg.batch_size + 1) * brats.test_batch), total=((155 // cfg.batch_size + 1) * brats.test_batch),
                          ncols=70, leave=False, unit='b'):
                img_org, label_2 = brats.next_test_batch(cfg.batch_size)
                feed_dict = {images: img_org}
                result = sess.run(net.result, feed_dict=feed_dict)

                result_color = utils.color_image(np.argmax(result[0], axis=2))
                if i % 50 == 0:
                    scp.misc.imsave(saveTestImgPath + brats.mhaName + str(i) + 'fcn_img.png', img_org[0])
                    # scp.misc.imsave(saveTestImgPath + str(i) + 'fcn_label.png', label_color)
                    scp.misc.imsave(saveTestImgPath + brats.mhaName + str(i) + 'fcn_result.png', result_color)
                brats.saveItk(np.argmax(result, axis=3),"multi_stage")
    pass

def main(_):

    tf.logging.info(' Start training...')
    lets_go()
    tf.logging.info('Training done')

if __name__ == "__main__":
    tf.app.run()