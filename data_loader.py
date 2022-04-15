import os
import numpy as np
import glob
import SimpleITK as sitk
import nibabel
from Fuzzy.config import cfg

class LOAD_BRATS_2015(object):

    def __init__(self):
        if cfg.year == 0:
            if cfg.computer == 0:
                if cfg.preprocess == 0:
                    if cfg.system == 0:
                        self.train_path_HGG = "E:/pycode/Alldataset/BRATS2015_Training/HGG/"
                        self.train_path_LGG = "E:/pycode/Alldataset/BRATS2015_Training/LGG/"
                        self.test_path = "E:/pycode/Alldataset/BRATS2015_Testing/HGG_LGG/"
                    if cfg.system == 1:
                        self.train_path_HGG = "/mnt/e/pycode/Alldataset/BRATS2015_Training/HGG/"
                        self.train_path_LGG = "/mnt/e/pycode/Alldataset/BRATS2015_Training/LGG/"
                        self.test_path = "/mnt/e/pycode/Alldataset/BRATS2015_Testing/HGG_LGG/"
                elif cfg.preprocess == 1:
                    self.train_path_HGG = "../MHA_change/N4_Normalized/HGG/"
                    self.train_path_LGG = "../MHA_change/N4_Normalized/LGG/"
                    self.test_path = "../MHA_change/N4_Normalized/test/"
                elif cfg.preprocess == 2:
                    self.train_path_HGG = "../MHA_change/N4_Normalized_correct/HGG/"
                    self.train_path_LGG = "../MHA_change/N4_Normalized_correct/LGG/"
                    self.test_path = "../MHA_change/N4_Normalized_correct/test/"
            elif cfg.computer == 1:
                self.train_path_HGG = "E:/BRATS2015/BRATS2015_Training/HGG/"
                self.train_path_LGG = "E:/BRATS2015/BRATS2015_Training/LGG/"
                self.test_path = "E:/BRATS2015/BRATS2015_Testing/HGG_LGG/"
            elif cfg.computer == 2:
                self.train_path_HGG = "E:/data/BRATS2015_Training/HGG/"
                self.train_path_LGG = "E:/data/BRATS2015_Training/LGG/"
                self.test_path = "E:/data/BRATS2015_Testing/HGG_LGG/"
        elif cfg.year == 1:
            if cfg.computer == 0:
                if cfg.preprocess == 0:
                    self.train_path_HGG = "E:/pycode/Alldataset/MICCAI_BraTS_2018_Data_Training/HGG/"
                    self.train_path_LGG = "E:/pycode/Alldataset/MICCAI_BraTS_2018_Data_Training/LGG/"
                    self.test_path = "E:/pycode/Alldataset/MICCAI_BraTS_2018_Data_Validation/"
                elif cfg.preprocess == 1:
                    self.train_path_HGG = "../MHA_change/2018/N4_Normalized/HGG/"
                    self.train_path_LGG = "../MHA_change/2018/N4_Normalized/LGG/"
                    self.test_path = "../MHA_change/2018/N4_Normalized/test/"
                elif cfg.preprocess == 2:
                    self.train_path_HGG = "../MHA_change/2018/N4_Normalized_correct/HGG/"
                    self.train_path_LGG = "../MHA_change/2018/N4_Normalized_correct/LGG/"
                    self.test_path = "../MHA_change/2018/N4_Normalized_correct/test/"
            elif cfg.computer == 1:
                self.train_path_HGG = "E:/BRATS2015/BRATS2015_Training/HGG/"
                self.train_path_LGG = "E:/BRATS2015/BRATS2015_Training/LGG/"
                self.test_path = "E:/BRATS2015/BRATS2015_Testing/HGG_LGG/"
            elif cfg.computer == 2:
                self.train_path_HGG = "E:/data/BRATS2015_Training/HGG/"
                self.train_path_LGG = "E:/data/BRATS2015_Training/LGG/"
                self.test_path = "E:/data/BRATS2015_Testing/HGG_LGG/"
        elif cfg.year == 2:
            if cfg.computer == 0:
                if cfg.preprocess == 0:
                    self.train_path_HGG = "E:/pycode/Alldataset/MICCAI_BraTS_2019_Data_Training/HGG/"
                    self.train_path_LGG = "E:/pycode/Alldataset/MICCAI_BraTS_2019_Data_Training/LGG/"
                    self.test_path = "E:/pycode/Alldataset/MICCAI_BraTS_2019_Data_Validation/"
                elif cfg.preprocess == 1:
                    self.train_path_HGG = "../MHA_change/2019/N4_Normalized/HGG/"
                    self.train_path_LGG = "../MHA_change/2019/N4_Normalized/LGG/"
                    self.test_path = "../MHA_change/2019/N4_Normalized/test/"
                elif cfg.preprocess == 2:
                    self.train_path_HGG = "../MHA_change/2019/N4_Normalized_correct/HGG/"
                    self.train_path_LGG = "../MHA_change/2019/N4_Normalized_correct/LGG/"
                    self.test_path = "../MHA_change/2019/N4_Normalized_correct/test/"
            elif cfg.computer == 1:
                self.train_path_HGG = "E:/BRATS2015/BRATS2015_Training/HGG/"
                self.train_path_LGG = "E:/BRATS2015/BRATS2015_Training/LGG/"
                self.test_path = "E:/BRATS2015/BRATS2015_Testing/HGG_LGG/"
            elif cfg.computer == 2:
                self.train_path_HGG = "E:/data/BRATS2015_Training/HGG/"
                self.train_path_LGG = "E:/data/BRATS2015_Training/LGG/"
                self.test_path = "E:/data/BRATS2015_Testing/HGG_LGG/"

    def load_file_list(self):
        self.OT_path = []
        self.Flair_path = []
        self.T1_path = []
        self.T1c_path = []
        self.T2_path = []
        self.id = []
        HGG_data = glob.glob(self.train_path_HGG + "*")
        LGG_data = glob.glob(self.train_path_LGG + "*")
        test_data = glob.glob(self.test_path + "*")
        HGG_data = [ x.replace("\\", "/") for x in HGG_data]
        LGG_data = [x.replace("\\", "/") for x in LGG_data]
        test_data = [x.replace("\\", "/") for x in test_data]
        if not cfg.is_testing:
            self.load_mod_list(HGG_data)
            if cfg.both_hgg_lgg:
                self.load_mod_list(LGG_data)
        else:
            self.load_mod_list(test_data)
        return self.OT_path, self.Flair_path, self.T1_path, self.T1c_path, self.T2_path, self.id

    def load_mod_list(self, data_path):
        for idx, file_name in enumerate(data_path):
            if cfg.year == 0:
                if cfg.preprocess == 0:
                    mod = glob.glob(file_name+"/*/*.mha*")
                elif cfg.preprocess == 1:
                    mod = glob.glob(file_name + "/*.mha*")
                elif cfg.preprocess == 2:
                    mod = glob.glob(file_name + "/*.mha*")
            elif cfg.year == 1:
                if cfg.preprocess == 0:
                    mod = glob.glob(file_name+"/*.nii*")
                elif cfg.preprocess == 1:
                    mod = glob.glob(file_name + "/*.nii*")
                elif cfg.preprocess == 2:
                    mod = glob.glob(file_name + "/*.nii*")
                if '.csv' not in file_name:
                    self.id.append(file_name.split('/')[-1])
            elif cfg.year == 2:
                pass

            mod = [x.replace("\\", "/") for x in mod]
            for mod_file in mod:
                # if 'Flair' not in mod_file and 'OT' not in mod_file:
                if 'OT' in mod_file or 'seg.' in mod_file:
                    self.OT_path.append(mod_file)
                if 'T1.' in mod_file or 't1.' in mod_file:
                    self.T1_path.append(mod_file)
                if 'T1c' in mod_file or 't1ce.' in mod_file:
                    self.T1c_path.append(mod_file)
                if 'T2' in mod_file or 't2.' in mod_file:
                    self.T2_path.append(mod_file)
                if 'Flair' in mod_file or 'flair.' in mod_file:
                    if cfg.year == 0:
                        self.id.append(mod_file.split('/')[-1])
                    self.Flair_path.append(mod_file)
        return None

    @staticmethod
    def load_from_file():
        brats = LOAD_BRATS_2015()
        return brats.load_file_list()

    @staticmethod
    def read_img( mha_path, isot=False):

        if cfg.year == 0:
            mha = sitk.ReadImage(mha_path)
            img_array = sitk.GetArrayFromImage(mha)
        else:
            img = nibabel.load(mha_path)
            img_array = np.transpose(img.get_data(), (2, 1, 0))
        if isot:
            return img_array
        else:
            # img_array = self.__repair(img_array)
            # 标准差标准化，经过处理的数据符合标准正态分布
            img_array = (img_array - img_array.mean())/ img_array.std()
            return img_array

            img_array = np.where(img_array > 0, img_array, 0)
            return img_array

            mask = img_array > 0
            mask = np.asarray(mask, dtype=np.float32)
            temp_img = img_array * mask
            return temp_img
            # temp_img = img_array[img_array > 0]
            # img_array = (img_array-temp_img.mean())/temp_img.std()
            # img_array = (img_array-img_array.min())/(img_array.max()-img_array.min())*mask
            img_array = (temp_img - temp_img.mean()) / temp_img.std()
            return img_array

    @staticmethod
    def read_img_test(mha_path, isot=False):
        if cfg.year == 0:
            mha = sitk.ReadImage(mha_path)
            img_array = sitk.GetArrayFromImage(mha)
        else:
            img = nibabel.load(mha_path)
            img_array = np.transpose(img.get_data(), (2, 1, 0))
        if isot:
            return img_array
        else:
            img_array = (img_array - img_array.mean()) / img_array.std()
            return img_array

    @staticmethod
    def postprocess_read_img_test(mha_path, isot=False, rgb=False):
        if cfg.year == 0:
            mha = sitk.ReadImage(mha_path)
            img_array = sitk.GetArrayFromImage(mha)
        else:
            img = nibabel.load(mha_path)
            img_array = np.transpose(img.get_data(), (2, 1, 0))
        if isot:
            if rgb:
                # img_array = np.abs(img_array)
                # 归一化到0-255
                y_max = 255
                y_min = 0
                max = np.max(img_array)
                min = np.min(img_array)
                # zhuyi yue jie!!!
                img_array = ((y_max - y_min)  / (max - min) ) * (img_array - min) + y_min

                #img_array = img_array /* max
            else:
                pass
            return img_array
        else:
            img_array = (img_array - img_array.mean()) / img_array.std()
            return img_array

    def __N4(self, input_image):

        # input_image = sitk.ReadImage(inputname)
        mask_image = sitk.OtsuThreshold(input_image, 0, 1, 200)
        input_image = sitk.Cast(input_image, sitk.sitkFloat32)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        output_image = corrector.Execute(input_image, mask_image)
        output_image = sitk.Cast(output_image, sitk.sitkInt16)
        # sitk.WriteImage(output_image, outputname)
        return output_image

    def __repair(self, img_batch):

        # # 归一化到1-255
        # y_max = 255
        # y_min = 1
        # max = np.max(img_array[img_array != 0])
        # min = np.min(img_array[img_array != 0])
        # img_array = np.where(img_array != 0, (y_max - y_min) * (img_array - min) / (max - min) + y_min, 0)

        assert img_batch.ndim == 3
        min_val = np.min(img_batch, axis=(0, 1, 2))
        if min_val == 0:
            return img_batch
        else:
            return np.abs(img_batch)
            return np.where(img_batch < 0, -img_batch, img_batch)


