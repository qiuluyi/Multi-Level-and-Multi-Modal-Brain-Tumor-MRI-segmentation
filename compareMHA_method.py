import SimpleITK as sitk
import numpy as np
from MHA_change.Region import Region
import os

class Compare:
    region1_avg = 0
    region2_avg = 0
    region3_avg = 0

    region1_ppv_avg = 0
    region2_ppv_avg = 0
    region3_ppv_avg = 0

    region1_s_avg = 0
    region2_s_avg = 0
    region3_s_avg = 0
    index = 0

    @staticmethod
    def compare(label_mha_path, up):
        Compare.index += 1
        mha = sitk.ReadImage(label_mha_path)
        label = sitk.GetArrayFromImage(mha)
        # print(result_path)
        dataR1, labelR1 = Region.Region1(up, label)
        region1 = dice_coef(dataR1, labelR1)
        Compare.region1_avg += region1
        region1_ppv = ppv_score(dataR1, labelR1)
        Compare.region1_ppv_avg += region1_ppv
        region1_s = sensitivity_score(dataR1, labelR1)
        Compare.region1_s_avg += region1_s

        dataR2, labelR2 = Region.Region2(up, label)
        region2 = dice_coef(dataR2, labelR2)
        Compare.region2_avg += region2
        region2_ppv = ppv_score(dataR2, labelR2)
        Compare.region2_ppv_avg += region2_ppv
        region2_s = sensitivity_score(dataR2, labelR2)
        Compare.region2_s_avg += region2_s
        #
        dataR3, labelR3 = Region.Region3(up, label)
        region3 = dice_coef(dataR3, labelR3)
        Compare.region3_avg += region3
        region3_ppv = ppv_score(dataR3, labelR3)
        Compare.region3_ppv_avg += region3_ppv
        region3_s = sensitivity_score(dataR3, labelR3)
        Compare.region3_s_avg += region3_s
        print('mha_name:\t', label_mha_path)
        print("region1:{} region2:{} region3:{}".format(region1, region2, region3))
        print("ppv: region1:{} region2:{} region3:{}".format(region1_ppv, region2_ppv, region3_ppv))
        print("sensitivity: region1:{} region2:{} region3:{}".format(region1_s, region2_s, region3_s))
        i = Compare.index
        print("index: " + str(i))
        print("dice: region1_avg:{} | region2_avg:{} | region3_avg:{}".format(Compare.region1_avg / i,
                                                                              Compare.region2_avg / i,
                                                                              Compare.region3_avg / i))
        print("ppv: region1_avg:{} | region2_avg:{} | region3_avg:{}".format(Compare.region1_ppv_avg / i,
                                                                             Compare.region2_ppv_avg / i,
                                                                             Compare.region3_ppv_avg / i))
        print("s: region1_avg:{} | region2_avg:{} | region3_avg:{}".format(Compare.region1_s_avg / i,
                                                                           Compare.region2_s_avg / i,
                                                                           Compare.region3_s_avg / i))

def dice_coef( y_true, y_pred):
    y_true_f = y_true
    y_pred_f = y_pred
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1)

def ppv_score( y_pred, y_true):
    y_true_f = y_true
    y_pred_f = y_pred
    TP = np.sum(y_true_f * y_pred_f)
    FP = np.sum((~y_true_f.astype("bool")) * y_pred_f)
    return (TP + 1) / (TP + FP + 1)

def sensitivity_score( y_pred, y_true):
    y_true_f = y_true
    y_pred_f = y_pred
    TP = np.sum(y_true_f * y_pred_f)
    FN = np.sum(y_true_f * (~y_pred_f.astype("bool")))
    # 0.9597406826322489
    return (TP + 1) / (TP + FN + 1)

# def sensitivity_score( y_pred, y_true):
#     y_true_f = y_true
#     y_pred_f = y_pred
#     TP = np.sum(y_true_f * y_pred_f)
#     #FN = np.sum(y_true_f * (~y_pred_f.astype("bool")))
#     # #0.9597406826322489
#     return (TP + 1) / (np.sum(y_true_f) + 1)



