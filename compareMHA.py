import SimpleITK as sitk
import numpy as np
from MHA_change.Region import Region
from Fuzzy.config import cfg

region1_avg = 0
region2_avg = 0
region3_avg = 0
i = 0


def dice_coef(y_true, y_pred):
    y_true_f = y_true
    y_pred_f = y_pred
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1)

import os

dimensional = ''
is_fuse = False
ground_truth_path = './mhaSavePath/mha_ground_truth/'
base_result_path = '/102901_015training_mha_'
base_result_path = '/102915_1_030training_mha__multi_stage'
base_result_path = '/011001c_015training_mha__multi_stage'
if is_fuse:
    result_path = './mhaSavePath' + base_result_path + dimensional + '_fuse'  + '/'
else:
    result_path = './mhaSavePath' + base_result_path + dimensional + '/'
ground_truth_paths = os.listdir(ground_truth_path)
result_paths = os.listdir(result_path)

# result_paths = os.listdir('./mhaSavePath/training_mha_fuse' + '/')
labels = []
ups = []
def sequence_inverted_interval_post_processing(data, interval=5, dimension=0):

    final = data.copy()
    num = data.shape[dimension]
    has_label = []
    for i in range(num):
        if dimension == 0:
            index_piece_sum = np.sum(data[i, :, :])
        elif dimension == 1:
            index_piece_sum = np.sum(data[:, i, :])
        else:
            index_piece_sum = np.sum(data[:, :, i])
        if index_piece_sum != 0:
            has_label.append(1)
        else:
            has_label.append(0)

    for idx in range(len(has_label)):
        if has_label[idx] == 1:
            if idx - interval < 0:
                interval_sum = np.sum(has_label[idx: idx + interval])
                if interval_sum != interval:
                    if dimension == 0:
                        final[idx, :, :] = 0
                    elif dimension == 1:
                        final[:, idx, :] = 0
                    else:
                        final[:,:,idx] = 0
            elif idx + interval > num:
                interval_sum = np.sum(has_label[idx - interval: idx])
                if interval_sum != interval:
                    if dimension == 0:
                        final[idx, :, :] = 0
                    elif dimension == 1:
                        final[:, idx, :] = 0
                    else:
                        final[:,:,idx] = 0
                pass
            else:
                interval_sum_begin = np.sum(has_label[idx: idx + interval])
                interval_sum_end = np.sum(has_label[idx - interval: idx])
                if interval_sum_begin != interval and interval_sum_end != interval:
                    if dimension == 0:
                        final[idx, :, :] = 0
                    elif dimension == 1:
                        final[:, idx, :] = 0
                    else:
                        final[:,:,idx] = 0
                pass
    return final

#result_paths = result_paths[1:]
for index in range(len(ground_truth_paths)):
    i += 1
    mha = sitk.ReadImage(ground_truth_path + ground_truth_paths[index])
    label = sitk.GetArrayFromImage(mha)
    labels.append(label)
    mha2 = sitk.ReadImage(result_path + result_paths[index])
    # mha2 = sitk.ReadImage('./mhaSavePath/training_mha_fuse' + '/'+ result_paths[index])
    up = sitk.GetArrayFromImage(mha2)
    #up = sequence_inverted_interval_post_processing(up, interval=17, dimension=1)
    #up = sequence_inverted_interval_post_processing(up, interval=15, dimension=2)
    #up = sequence_inverted_interval_post_processing(up, interval=17, dimension=0)

    ups.append(up)

    # print(result_path)
    dataR1, labelR1 = Region.Region1(up, label)
    region1 = dice_coef(dataR1, labelR1)
    region1_avg += region1

    dataR2, labelR2 = Region.Region2(up, label)
    region2 = dice_coef(dataR2, labelR2)
    region2_avg += region2

    #
    dataR3, labelR3 = Region.Region3(up, label)
    region3 = dice_coef(dataR3, labelR3)

    region3_avg += region3
    print('mha_name:\t',ground_truth_paths[index])
    print("region1:{} region2:{} region3:{}".format(region1,region2,region3))
print('i: {} \t lens: {}'.format(i,len(ground_truth_paths)))
print("region1_avg:{} region2_avg:{} region3_avg:{}".format(region1_avg/i, region2_avg/i, region3_avg/i))

# ups = np.array(ups)
# labels = np.array(labels)
#
# datatotle1, labeltotle1 = Region.Region1(ups, labels)
# regiontotle1 = dice_coef(datatotle1, labeltotle1)
# print(regiontotle1)
# datatotle2, labeltotle2 = Region.Region2(ups, labels)
# regiontotle2 = dice_coef(datatotle2, labeltotle2)
# print(regiontotle2)
#
# datatotle3, labeltotle3 = Region.Region3(ups, labels)
# regiontotle3 = dice_coef(datatotle3, labeltotle3)
# print(regiontotle3)

    # print(up.shape)
    # dataR1, labelR1 = Region.Region1(up, label)
    # region1 = dice_coef(dataR1, labelR1)
	#
    # # dataR1, labelR1 = Region.Region1(up, label)
    # # region1 = dice_coef(dataR1, labelR1)
    # dataR2, labelR2 = Region.Region2(up, label)
    # region2 = dice_coef(dataR2, labelR2)
    # #
    # dataR3, labelR3 = Region.Region3(up, label)
    # region3 = dice_coef(dataR3, labelR3)
	#
    # print("region1:\t",region1)
    # print("region2:\t",region2)
    # print("region3:\t",region3)

    # print(len(labels))

# datatotle1, labeltotle1 = Region.Region1(ups, labels)
# regiontotle1 = dice_coef(datatotle1, labeltotle1)
#
# datatotle2, labeltotle2 = Region.Region2(ups, labels)
# regiontotle2 = dice_coef(datatotle2, labeltotle2)
#
# datatotle3, labeltotle3 = Region.Region3(ups, labels)
# regiontotle3 = dice_coef(datatotle3, labeltotle3)
#
# print(regiontotle1+'\t'+regiontotle2+'\t'+regiontotle3)

    # dataR1, labelR1 = Region.Region1(up, label)
    # region1 = dice_coef(dataR1, labelR1)
    # dataR2, labelR2 = Region.Region2(up, label)
    # region2 = dice_coef(dataR2, labelR2)
    # #
    # dataR3, labelR3 = Region.Region3(up, label)
    # region3 = dice_coef(dataR3, labelR3)


    # print(region1)
    # print(region2)
    # print(region3)
# for i in range(155):
#     print("test:" + str(i))
#     dataR1, labelR1 = Region.Region1(up[i], label[i])
#     region1 = dice_coef(dataR1, labelR1)
#     region1_avg += region1
#
#     dataR2, labelR2 = Region.Region2(up[i], label[i])
#     region2 = dice_coef(dataR2, labelR2)
#     region2_avg += region2
#
#     dataR3, labelR3 = Region.Region3(up[i], label[i])
#     region3 = dice_coef(dataR3, labelR3)
#     region3_avg += region3
#
#     print("Region1:", region1 )
#     print("Region2:", region2 )
#     print("Region3:", region3 )
#
# print("Region1:", region1_avg/155)
# print("Region2:", region2_avg/155)
# print("Region3:", region3_avg/155)
#
