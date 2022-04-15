import os
import SimpleITK as sitk
import numpy as np
import re
import shutil
import nibabel
from Fuzzy.data_loader import LOAD_BRATS_2015
from Fuzzy.config import cfg
from Fuzzy.compareMHA_method import Compare

def makelabel_tf(label):
    label = np.reshape(label,[np.shape(label)[0], -1])
    l =np.max(label,axis=1)
    one = np.asarray([1 if i > 0 else 0 for i in l])
    return one

class BRATS2015:
    def __init__(self, train_batch_count=0, test_batch_count=0, train=True, test=False, both_hgg_lgg=True, validation_set_num=0):

        self.train_batch_count = train_batch_count
        self.test_batch_count = test_batch_count
        self.train = train
        self.test = test
        self.OT_path, self.Flair_path, self.T1_path, self.T1c_path, self.T2_path, self.id = LOAD_BRATS_2015.load_from_file()

        if test:
            self.total_batch = len(self.T1_path)
            self.test_batch = self.total_batch
            self.next_test_MHA()
            self.test_batch_index = 0
        else:
            self.total_batch = len(self.OT_path)
            self.test_batch = validation_set_num
            self.train_batch = self.total_batch - self.test_batch
            if train:
                self.next_train_MHA()
                self.train_batch_index = 0
            else:
                self.next_test_MHA()
                self.test_batch_index = 0
        self.saveArr = None

    def next_test_MHA(self):

        self.test_batch_count = self.test_batch_count % self.test_batch
        if self.test:
            test_index = self.test_batch_count
        else:
            test_index = self.test_batch_count + self.train_batch
            ot = self.OT_path[test_index]
            print('VSD.InstanceFCN_' + re.findall(r"\.\d+\.", ot)[0] + 'mha')
            img_array = LOAD_BRATS_2015.read_img_test(ot, isot=True)
            arr_ot = np.asarray(img_array)
            self.arr_test_ot = arr_ot  # .reshape(num, height, weight)

        t1 = self.T1_path[test_index]
        t2 = self.T2_path[test_index]
        t1c = self.T1c_path[test_index]
        flair = self.Flair_path[test_index]

        if cfg.year != 0:
            idName = flair.split('/')
            self.mhaName = idName[-1][0: idName[-1].rfind('_')]
        else:
            self.mhaName = re.findall(r"\.\d+\.", flair)[0]

        img_array_t1 = LOAD_BRATS_2015.read_img_test(t1)
        img_array_t2 = LOAD_BRATS_2015.read_img_test(t2)
        img_array_t1c = LOAD_BRATS_2015.read_img_test(t1c)
        img_array_flair = LOAD_BRATS_2015.read_img_test(flair)

        # num, height, weight = np.shape(arr_ot)
        # arr_t1 = np.expand_dims(img_array_t1, -1) #np.asarray(img_array_t1).reshape(num, height, weight, 1)
        # arr_t1c = np.expand_dims(img_array_t1c, -1) #np.asarray(img_array_t1c).reshape(num, height, weight, 1)
        # arr_t2 = np.expand_dims(img_array_t2, -1) #np.asarray(img_array_t2).reshape(num, height, weight, 1)
        # arr_flair = np.expand_dims(img_array_flair, -1) #np.asarray(img_array_flair).reshape(num, height, weight, 1)

        #self.arr_test_imgs = np.stack([img_array_flair, img_array_t1c, img_array_t2], axis=3)
        self.arr_test_imgs = np.stack([img_array_flair, img_array_t1, img_array_t1c, img_array_t2], axis=3)

        # self.arr_test_imgs = np.concatenate((arr_flair, arr_t1, arr_t1c, arr_t2), axis=3)
        # shape = self.arr_test_imgs.shape
        # self.arr_test_imgs = self.arr_test_imgs.reshape(-1, shape[3])
        # self.standard_scaler.fit(self.arr_train_imgs)
        # Max_min = preprocessing.MinMaxScaler()
        # self.arr_test_imgs = preprocessing.StandardScaler().fit_transform(self.arr_test_imgs)
        # self.arr_test_imgs = self.arr_test_imgs.reshape(shape[0], shape[1], shape[2], shape[3])

        self.test_batch_count += 1

    def next_test_batch(self, batch_size):
        endbatch = np.shape(self.arr_test_imgs)[0]
        # print(self.test_batch_index + batch_size, endbatch)
        if self.test_batch_index + batch_size >= endbatch:
            img = self.arr_test_imgs[self.test_batch_index:endbatch]
            if not self.test:
                label = self.arr_test_ot[self.test_batch_index:endbatch]
            else:
                label = np.asarray(0)
            self.test_batch_index = 0
            self.next_test_MHA()
        else:
            img = self.arr_test_imgs[self.test_batch_index:self.test_batch_index + batch_size]
            if not self.test:
                label = self.arr_test_ot[self.test_batch_index:self.test_batch_index + batch_size]
            else:
                label = np.asarray(0)
            self.test_batch_index += batch_size
        # label = np.eye(5)[label]
        # label[:, :, :, 0] = label[:, :, :, 0] * 0.05
        return img, label  # , makelabel_tf(label)
            # return img, label

    def next_train_MHA(self):
        # if self.train_batch_count > self.train_batch:

        self.train_batch_count = self.train_batch_count % self.train_batch

        # print(self.OT_path[self.train_batch_count])
        ot = self.OT_path[self.train_batch_count]
        t1 =  self.T1_path[self.train_batch_count]
        t2 = self.T2_path[self.train_batch_count]
        t1c = self.T1c_path[self.train_batch_count]
        flair = self.Flair_path[self.train_batch_count]
        # print('VSD.InstanceFCN_' + re.findall(r"\.\d+\.", ot)[0] + 'mha')

        if cfg.year != 0:
            idName = flair.split('/')
            self.mhaName = idName[-1][0: idName[-1].rfind('_')]
        else:
            self.mhaName = re.findall(r"\d+\.", ot)[0]

        img_array = LOAD_BRATS_2015.read_img(ot, isot=True)
        img_array_t1 = LOAD_BRATS_2015.read_img(t1)
        img_array_t2 = LOAD_BRATS_2015.read_img(t2)
        img_array_t1c = LOAD_BRATS_2015.read_img(t1c)
        img_array_flair = LOAD_BRATS_2015.read_img(flair)

        arr_ot = np.asarray(img_array)
        # num, height, weight, chanel = np.shape(arr_ot)

        # index = np.asarray(range(num))
        # np.random.shuffle(index)
        #
        # arr_ot = arr_ot[index]
        # img_array_t1 = img_array_t1[index]
        # img_array_t2 = img_array_t2[index]
        # img_array_flair = img_array_flair[index]
        # img_array_t1c = img_array_t1c[index]
        #
        self.arr_train_ot = arr_ot #.reshape(num, height, weight)
        # # self.arr_train_ot[self.arr_train_ot == 0] = -1
        # arr_t1 = np.expand_dims(img_array_t1, -1)#np.asarray(img_array_t1).reshape(num, height, weight, 1)
        # arr_t1c = np.expand_dims(img_array_t1c, -1)#np.asarray(img_array_t1c).reshape(num, height, weight, 1)
        # arr_t2 = np.expand_dims(img_array_t2, -1)#np.asarray(img_array_t2).reshape(num, height, weight, 1)
        # arr_flair = np.expand_dims(img_array_flair, -1)#np.asarray(img_array_flair).reshape(num, height, weight, 1)
        # self.arr_train_imgs = np.concatenate((arr_flair, arr_t1, arr_t1c, arr_t2), axis=3)

        #self.arr_train_imgs = np.stack([img_array_flair, img_array_t1c, img_array_t2], axis=3)
        self.arr_train_imgs = np.stack([img_array_flair, img_array_t1, img_array_t1c, img_array_t2], axis=3)

        # shape = self.arr_train_imgs.shape
        # self.arr_train_imgs = self.arr_train_imgs.reshape(-1, shape[3])
        # self.arr_train_imgs = self.arr_train_imgs
        # self.arr_train_imgs = self.arr_train_imgs.reshape([shape[0], shape[1], shape[2], shape[3]])

        # 只训练有肿瘤的部分
        # has = np.max(self.arr_train_ot, axis=(1, 2)) > 0
        # self.arr_train_ot = self.arr_train_ot[has]
        # self.arr_train_imgs = self.arr_train_imgs[has]

        # index = np.asarray(range(len(self.arr_train_imgs)))
        # np.random.shuffle(index)
        # # print(index)
        # self.arr_train_ot[index]
        # self.arr_train_imgs[index]

        # print(self.arr_train_imgs.shape)
        self.train_batch_count += 1

    def next_train_batch(self, batch_size):
        endbatch = np.shape(self.arr_train_ot)[0]
        # print(self.train_batch_index + batch_size, endbatch)
        if self.train_batch_index + batch_size >= endbatch:
            img = self.arr_train_imgs[self.train_batch_index:endbatch]
            label = self.arr_train_ot[self.train_batch_index:endbatch]
            self.train_batch_index = 0
            self.next_train_MHA()
        else:
            img = self.arr_train_imgs[self.train_batch_index:self.train_batch_index + batch_size]
            label = self.arr_train_ot[self.train_batch_index:self.train_batch_index + batch_size]
            self.train_batch_index += batch_size

        # label = np.eye(5)[label]
        # label[:,:,:,0] = label[:,:,:,0] * 0.01
        # class_weight[:,:,:,0] = class_weight[:,:,:,0] * 0.1
        return img, label

    def next_train_patient(self):

        img = self.arr_train_imgs
        label = self.arr_train_ot
        self.next_train_MHA()
        return img, label

    def saveItk(self, array, name):
        self.compareMHA_index = 0
        array = np.asarray(array)
        if self.saveArr is not None:
            self.saveArr = np.concatenate([self.saveArr, array], axis=0)
        else:
            self.saveArr = array
        if self.test_batch_index == 0:
            # print(np.shape(self.saveArr))
            # not need
            # if cfg.year != 0:
            #     OUTPUT_AFFINE = np.array(
            #         [[0, 0, 1, 0],
            #          [0, 1, 0, 0],
            #          [1, 0, 0, 0],
            #          [0, 0, 0, 1]])
            #     img = nibabel.Nifti1Image(self.saveArr.astype(np.uint8), OUTPUT_AFFINE)
            # else:
            img = sitk.GetImageFromArray(self.saveArr)
            mhaGroundTruthPath = 'mhaSavePath/mha_ground_truth/'
            if self.test:
                mhaIsTest = 'something_0606_20_035testing'
                #mhaIsTest = 'unet_1029_13_3_030testing'
                mhaIsTest = 'unet_210105_05c_1_030testing'
                #mhaIsTest = 'unet_201020_01a_7_030testing'
                #mhaIsTest = 'unet_200121_03_030testing'
                #mhaIsTest = 'slicer_20190715_01'
                # Reverse order no problem
                path = self.id[self.test_batch_count - 2]
            else:
                mhaIsTest = '102915_1_030training'
                mhaIsTest = '011001c_015training'
                mhaIsTest = '20210105_01c_2_001training'
                if self.test_batch_count == 0:
                    path = self.OT_path[self.test_batch + self.train_batch - 2]
                elif self.test_batch_count == 1:
                    path = self.OT_path[self.test_batch + self.train_batch - 1]
                else:
                    path = self.OT_path[self.test_batch_count + self.train_batch - 2]
                truth_name = 'VSD.InstanceFCN_Truth' + re.findall(r"\.\d+\.", path)[0] + 'mha'
                if not os.path.exists(mhaGroundTruthPath):
                    os.makedirs(mhaGroundTruthPath)
                mhaGroundTruthFile = os.path.abspath(mhaGroundTruthPath) + "/" + truth_name
                if not os.path.exists(mhaGroundTruthFile):
                    shutil.copy(path, mhaGroundTruthFile)

            # directly compare
            if False:
                ground_truth_paths = os.listdir(mhaGroundTruthPath)
                Compare.compare(path, self.saveArr)
            else:
                if cfg.year == 0:
                    save_name = 'VSD.InstanceFCN_' + name + re.findall(r"\.\d+\.", path)[0] + 'mha'
                else:
                    save_name = path + '.nii.gz'
                mhaTrainFuseSavePath = 'mhaSavePath/' + mhaIsTest + '_mha_' + '_' + name + '/'
                if not os.path.exists(mhaTrainFuseSavePath):
                    os.makedirs(mhaTrainFuseSavePath)
                sitk.WriteImage(sitk.Cast(img, sitk.sitkUInt8), os.path.join(mhaTrainFuseSavePath, save_name))
            self.saveArr = None


    def save_train_Itk(self, array):
        array = np.asarray(array)
        if self.saveArr is not None:
            self.saveArr = np.concatenate([self.saveArr, array], axis=0)
        else:
            self.saveArr = array
        if self.train_batch_index == 0:
            # print(np.shape(self.saveArr))
            img = sitk.GetImageFromArray(self.saveArr)
            path = self.OT_path[self.train_batch_count - 2]
            name = 'VSD.InstanceFCN' + re.findall(r"\.\d+\.", path)[0] + 'mha'
            shutil.copy(path, 'mha_ground_truth/' + name)
            sitk.WriteImage(sitk.Cast(img, sitk.sitkUInt8), os.path.join('mhaSavePath/', name), )
            self.saveArr = None

if False:
    brats = BRATS2015(10)
    # i = brats.arr_train_imgs#= brats.next_train_batch(1)
    # image, label = brats.next_train_batch(5)
    import matplotlib.pyplot as plt

    for i in range(50):
        image, label = brats.next_train_batch(5)
        for nnn in range(4):
            i1 = image[nnn, :, :, 0]
            sub = plt.subplot(231)
            sub.imshow(i1, cmap='gray')
            i1 = image[nnn, :, :, 1]
            sub = plt.subplot(232)
            sub.imshow(i1, cmap='gray')
            i1 = image[nnn, :, :, 2]
            sub = plt.subplot(233)
            sub.imshow(i1, cmap='gray')
            i1 = image[nnn, :, :, 3]
            sub = plt.subplot(234)
            sub.imshow(i1, cmap='gray')
            i1 = label[nnn, :, :]
            sub = plt.subplot(235)
            sub.imshow(i1, cmap='gray')
            i1 = label[nnn, :, :]
            sub = plt.subplot(236)
            sub.imshow(i1, cmap='gray')
            plt.show()

# a = []
# for i in range(190):
#    l = brats.arr_train_ot
#    print(np.sum(l,(0,1,2)))
#    l = np.eye(5)[l]
#    # print(np.sum(l,(3)))
#    print(np.sum(l,(0,1,2)))
#    a.append(np.sum(l, (0,1,2)))
#    brats.next_train_MHA()
#
# a = np.asarray(a)
# print(a.sum(0))

#
# label_to_frequency =  [  1.67481864e+09,   1.05296600e+06,   1.39822550e+07,   2.25209600e+06,   4.21404400e+06]
# class_weights = []
#
# total_frequency = np.sum(label_to_frequency)
# for frequency in label_to_frequency:
#     class_weight = 1 / np.log(1.02 + (frequency / total_frequency))
#     class_weights.append(class_weight)
# print(class_weights)

# [1.4496326123173364, 47.438213320786453, 27.297160306177677, 44.379722047746604, 40.152262287301141]
