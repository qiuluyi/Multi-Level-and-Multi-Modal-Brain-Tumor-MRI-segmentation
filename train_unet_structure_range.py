import logging
import sys
import tensorflow as tf
import scipy as scp
import os
import numpy as np
import cv2 as cv

# from MHA_change.readMHATesting import BRATS2015Test as Brats
from Fuzzy.readMHA_something import BRATS2015 as Brats
from Fuzzy.model_unet_structure_range import model
from MHA_change.Region import Region
from MHA_change.utils import utils
from Fuzzy.config import cfg
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

# Parse Options
path = 'RESNEXT_unet_20191217_02' + cfg.dimensional + '/'
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
        fd_train_data.write('global_step, loss, acc, dice, dice2, dice3, all_loss_show\n')
        return(fd_train_data)

ra = np.random.randint(0, 274) # ra = np.random.randint(0, 274)  # random to choose the begining brats
brats = Brats(8,0,train = cfg.is_training,test=cfg.is_testing, both_hgg_lgg=cfg.both_hgg_lgg, validation_set_num=cfg.validation_set_num)
# load_npy = load_from_npy()
def count_flops(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    print('FLOPs: {}'.format(flops.total_float_ops))

def lets_go():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        images = tf.placeholder("float32", [None, 240, 240, 4])
        outputs = tf.placeholder("float16", [None, 240, 240])
        net = model(class_num=1, img=images, out=outputs)  # ,train=train_able
        # count_flops(sess.graph)
        # is_flip_image = tf.placeholder("bool",False)
        # net = model(is_flip=is_flip_image, class_num=5, img=images, out=outputs)  # ,train=train_able
        print('build network success')
        all_loss = tf.get_collection("loss")
        init = tf.global_variables_initializer()
        sess.run(init)
        # tf.image.resize_images()
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
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
                    img, label = brats.next_train_batch(cfg.batch_size)
                    label = np.where(label == 0, 0.1,
                             np.where(label == 1, 0.3,
                                      np.where(label == 2, 0.5,
                                               np.where(label == 3, 0.7,
                                                            0.9
                                                        )
                                               )
                                      )
                             )

                    #label[label==3] = 1
                    # if epoch % 2 != 0:
                    #     model.is_flip = True
                    # else:
                    #     model.is_flip = False
                    feed_dict = {images: img, outputs: label}
                    sess.run(net.train_op, feed_dict=feed_dict)
                    if i % 69 == 0:
                        summary, loss, result, global_step, label_min, all_loss_show = sess.run([merged,
                                                                                                      net.loss,
                                                                                                      net.result,
                                                                                                      net.global_step,
                                                                                                      net.output,
                                                                                                      all_loss
                                                                                                      ],
                                                                                                     feed_dict=feed_dict)


                        result = np.where(result <= 0.2, 0,
                                         np.where(result <= 0.4, 1,
                                                  np.where(result <= 0.6, 2,
                                                           np.where(result <= 0.8, 3,
                                                                    4
                                                                    )
                                                           )
                                                  )
                                         )

                        label_min = np.where(label_min <= 0.2, 0,
                                         np.where(label_min <= 0.4, 1,
                                                  np.where(label_min <= 0.6, 2,
                                                           np.where(label_min <= 0.8, 3,
                                                                    4
                                                                    )
                                                           )
                                                  )
                                         )
                        train_writer.add_summary(summary, global_step)
                        dataR1, labelR1 = Region.Region1(result, label_min)
                        dice = utils.dice_coef(dataR1, labelR1)
                        dataR2, labelR2 = Region.Region2(result, label_min)
                        dice2 = utils.dice_coef(dataR2, labelR2)
                        dataR3, labelR3 = Region.Region3(result, label_min)
                        dice3 = utils.dice_coef(dataR3, labelR3)

                        # print Image()
                        label_color = utils.color_image(label_min[0, :, :])
                        result_color = utils.color_image(result[0])
                        scp.misc.imsave(saveImgPath + brats.mhaName + str(i) + 'fcn_img.png', img[0])
                        scp.misc.imsave(saveImgPath + brats.mhaName + str(i) + 'fcn_label.png', label_color)
                        scp.misc.imsave(saveImgPath + brats.mhaName + str(i) + 'fcn_result.png', result_color)

                        # print(global_step, loss, acc, dice, dice2, dice3, all_loss_show)
                        fd_train_data.write(str(global_step) + ',' + str(loss) + ','
                                            + str(dice) + ',' + str(dice2) + ',' + str(dice3) + ',' + str(
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
                img, label = brats.next_test_batch(cfg.batch_size)
                feed_dict = {images: img, outputs: label}
                result = sess.run(net.logits, feed_dict=feed_dict)
                result_color = utils.color_image(np.argmax(result[0], axis=2))
                if i % 50 == 0:
                    scp.misc.imsave(saveTestImgPath + brats.mhaName + str(i) + 'fcn_img.png', img[0])
                    # scp.misc.imsave(saveTestImgPath + str(i) + 'fcn_label.png', label_color)
                    scp.misc.imsave(saveTestImgPath + brats.mhaName + str(i) + 'fcn_result.png', result_color)
                brats.saveItk(np.argmax(result, axis=3))
        else:
            for i in tqdm(range((155 // cfg.batch_size + 1) * brats.test_batch), total=((155 // cfg.batch_size + 1) * brats.test_batch),
                          ncols=70, leave=False, unit='b'):
                img, label = brats.next_test_batch(cfg.batch_size)
                feed_dict = {images: img}
                result = sess.run(net.logits, feed_dict=feed_dict)
                result_color = utils.color_image(np.argmax(result[0], axis=2))
                if i % 50 == 0:
                    scp.misc.imsave(saveTestImgPath + brats.mhaName + str(i) + 'fcn_img.png', img[0])
                    # scp.misc.imsave(saveTestImgPath + str(i) + 'fcn_label.png', label_color)
                    scp.misc.imsave(saveTestImgPath + brats.mhaName + str(i) + 'fcn_result.png', result_color)
                brats.saveItk(np.argmax(result, axis=3),"symmetry")
    pass

def main(_):

    tf.logging.info(' Start training...')
    lets_go()
    tf.logging.info('Training done')

if __name__ == "__main__":
    tf.app.run()