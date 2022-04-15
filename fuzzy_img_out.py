import cv2 as cv
from Fuzzy.readMHA_something import BRATS2015 as Brats
import matplotlib.pyplot as plt
import scipy as scp
import os
import numpy as np

brats = Brats(8)
savepath = 'fuzzy_img_out/'
# 16+31*10
# 16+31*15
# 16+31*16

for i in range(16+31*15):
    image, label = brats.next_train_batch(5)

image_f = image[:, :, :, 0]
image_t1 = image[:, :, :, 1]
image_t1c = image[:, :, :, 2]
image_t2 = image[:, :, :, 3]
# kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
# dilation = np.asarray(image[:, :, :, 2].copy(), dtype=np.uint8)
# for j in range(image.shape[0]):
#     dilation[j, :, :] = cv.dilate(dilation[j, :, :], kernel)

img_temp_f_1 = image_f.copy()
img_temp_f_2 = image_f.copy()
img_temp_f_3 = image_f.copy()
img_temp_f_4 = image_f.copy()

img_temp_t1_1 = image_t1.copy()
img_temp_t1_2 = image_t1.copy()
img_temp_t1_3 = image_t1.copy()
img_temp_t1_4 = image_t1.copy()

img_temp_t1c_1 = image_t1c.copy()
img_temp_t1c_2 = image_t1c.copy()
img_temp_t1c_3 = image_t1c.copy()
img_temp_t1c_4 = image_t1c.copy()

img_temp_t2_1 = image_t2.copy()
img_temp_t2_2 = image_t2.copy()
img_temp_t2_3 = image_t2.copy()
img_temp_t2_4 = image_t2.copy()

# dilation = np.expand_dims(dilation, axis=-1)
# img_temp = np.where(dilation > 0, 0, img_temp)

# for k in range(img_temp_f_1.shape[0]):
#     img_temp_f_1[k, :, :] = cv.GaussianBlur(img_temp_f_1[k, :, :], (0, 0), 1)
#     img_temp_t1_1[k, :, :] = cv.GaussianBlur(img_temp_t1_1[k, :, :], (0, 0), 1)
#     img_temp_t1c_1[k, :, :] = cv.GaussianBlur(img_temp_t1c_1[k, :, :], (0, 0), 1)
#     img_temp_t2_1[k, :, :] = cv.GaussianBlur(img_temp_t2_1[k, :, :], (0, 0), 1)
# for k in range(img_temp_f_2.shape[0]):
#     img_temp_f_2[k, :, :] = cv.GaussianBlur(img_temp_f_2[k, :, :], (0, 0), 3)
#     img_temp_t1_2[k, :, :] = cv.GaussianBlur(img_temp_t1_2[k, :, :], (0, 0), 3)
#     img_temp_t1c_2[k, :, :] = cv.GaussianBlur(img_temp_t1c_2[k, :, :], (0, 0), 3)
#     img_temp_t2_2[k, :, :] = cv.GaussianBlur(img_temp_t2_2[k, :, :], (0, 0), 3)
# for k in range(img_temp_f_3.shape[0]):
#     img_temp_f_3[k, :, :] = cv.GaussianBlur(img_temp_f_3[k, :, :], (0, 0), 5)
#     img_temp_t1_3[k, :, :] = cv.GaussianBlur(img_temp_t1_3[k, :, :], (0, 0), 5)
#     img_temp_t1c_3[k, :, :] = cv.GaussianBlur(img_temp_t1c_3[k, :, :], (0, 0), 5)
#     img_temp_t2_3[k, :, :] = cv.GaussianBlur(img_temp_t2_3[k, :, :], (0, 0), 5)
# for k in range(img_temp_f_4.shape[0]):
#     img_temp_f_4[k, :, :] = cv.GaussianBlur(img_temp_f_4[k, :, :], (0, 0), 7)
#     img_temp_t1_4[k, :, :] = cv.GaussianBlur(img_temp_t1_4[k, :, :], (0, 0), 7)
#     img_temp_t1c_4[k, :, :] = cv.GaussianBlur(img_temp_t1c_4[k, :, :], (0, 0), 7)
#     img_temp_t2_4[k, :, :] = cv.GaussianBlur(img_temp_t2_4[k, :, :], (0, 0), 7)

for k in range(img_temp_f_1.shape[0]):
    img_temp_f_1[k, :, :] = cv.GaussianBlur(img_temp_f_1[k, :, :], (7, 7),0)
    img_temp_t1_1[k, :, :] = cv.GaussianBlur(img_temp_t1_1[k, :, :], (7, 7), 0)
    img_temp_t1c_1[k, :, :] = cv.GaussianBlur(img_temp_t1c_1[k, :, :], (7, 7), 0)
    img_temp_t2_1[k, :, :] = cv.GaussianBlur(img_temp_t2_1[k, :, :], (7, 7), 0)
for k in range(img_temp_f_2.shape[0]):
    img_temp_f_2[k, :, :] = cv.GaussianBlur(img_temp_f_2[k, :, :], (11, 11), 0)
    img_temp_t1_2[k, :, :] = cv.GaussianBlur(img_temp_t1_2[k, :, :], (11, 11), 0)
    img_temp_t1c_2[k, :, :] = cv.GaussianBlur(img_temp_t1c_2[k, :, :], (11, 11), 0)
    img_temp_t2_2[k, :, :] = cv.GaussianBlur(img_temp_t2_2[k, :, :], (11, 11), 0)
for k in range(img_temp_f_3.shape[0]):
    img_temp_f_3[k, :, :] = cv.GaussianBlur(img_temp_f_3[k, :, :], (15, 15), 0)
    img_temp_t1_3[k, :, :] = cv.GaussianBlur(img_temp_t1_3[k, :, :], (15, 15), 0)
    img_temp_t1c_3[k, :, :] = cv.GaussianBlur(img_temp_t1c_3[k, :, :], (15, 15), 0)
    img_temp_t2_3[k, :, :] = cv.GaussianBlur(img_temp_t2_3[k, :, :], (15, 15), 0)
for k in range(img_temp_f_4.shape[0]):
    img_temp_f_4[k, :, :] = cv.GaussianBlur(img_temp_f_4[k, :, :], (19, 19), 0)
    img_temp_t1_4[k, :, :] = cv.GaussianBlur(img_temp_t1_4[k, :, :], (19, 19), 0)
    img_temp_t1c_4[k, :, :] = cv.GaussianBlur(img_temp_t1c_4[k, :, :], (19, 19), 0)
    img_temp_t2_4[k, :, :] = cv.GaussianBlur(img_temp_t2_4[k, :, :], (19, 19), 0)

if not os.path.exists(savepath):
    os.makedirs(savepath)

for nnn in range(4):
    if np.max(label)>0:

        img_f = np.concatenate((image[nnn, :, :, 0], img_temp_f_1[nnn, :, :], img_temp_f_2[nnn, :, :], img_temp_f_3[nnn, :, :], img_temp_f_4[nnn, :, :]),
                                 axis=0)
        img_t1 = np.concatenate((image[nnn, :, :, 1],img_temp_t1_1[nnn, :, :], img_temp_t1_2[nnn, :, :], img_temp_t1_3[nnn, :, :], img_temp_t1_4[nnn, :, :]),
                                 axis=0)
        img_t1c = np.concatenate((image[nnn, :, :, 2],img_temp_t1c_1[nnn, :, :], img_temp_t1c_2[nnn, :, :], img_temp_t1c_3[nnn, :, :], img_temp_t1c_4[nnn, :, :]),
                                 axis=0)
        img_t2 = np.concatenate((image[nnn, :, :, 3],img_temp_t2_1[nnn, :, :], img_temp_t2_2[nnn, :, :], img_temp_t2_3[nnn, :, :], img_temp_t2_4[nnn, :, :]),
                                 axis=0)


        scp.misc.imsave(savepath + brats.mhaName + '_' + str(nnn) + '_'+ 'f_img.png', img_f)

        scp.misc.imsave(savepath + brats.mhaName + '_' + str(nnn) + '_'+ 't1_img.png', img_t1)

        scp.misc.imsave(savepath + brats.mhaName + '_' + str(nnn) + '_'+ 't1c_img.png', img_t1c)

        scp.misc.imsave(savepath + brats.mhaName + '_' + str(nnn) + '_'+ 't2_img.png', img_t2)

        # img_org = np.concatenate((image[nnn, :, :, 0], image[nnn, :, :, 1], image[nnn, :, :, 2], image[nnn, :, :, 3]), axis=1)
        # img_f = np.concatenate((img_temp_f_1[nnn, :, :], img_temp_t1_1[nnn, :, :], img_temp_t1c_1[nnn, :, :], img_temp_t2_1[nnn, :, :]),
        #                          axis=1)
        # img_t1 = np.concatenate((img_temp_f_2[nnn, :, :], img_temp_t1_2[nnn, :, :], img_temp_t1c_2[nnn, :, :], img_temp_t2_2[nnn, :, :]),
        #                          axis=1)
        # img_t1c = np.concatenate((img_temp_f_3[nnn, :, :], img_temp_t1_3[nnn, :, :], img_temp_t1c_3[nnn, :, :], img_temp_t2_3[nnn, :, :]),
        #                          axis=1)
        # img_t2 = np.concatenate((img_temp_f_4[nnn, :, :], img_temp_t1_4[nnn, :, :], img_temp_t1c_4[nnn, :, :], img_temp_t2_4[nnn, :, :]),
        #                          axis=1)
        # img_final = np.concatenate((img_org, img_f, img_t1, img_t1c, img_t2),
        #                          axis=0)
        # scp.misc.imsave(brats.mhaName + '_' + str(nnn) + '_'+ 'fuzzy_img.png', img_final)

# for nnn in range(1):
#     if np.max(label)>0:
#         i1 = image[nnn, :, :, 0]
#         sub = plt.subplot(231)
#         sub.imshow(i1, cmap='gray')
#         i1 = image[nnn, :, :, 1]
#         sub = plt.subplot(232)
#         sub.imshow(i1, cmap='gray')
#         i1 = image[nnn, :, :, 2]
#         sub = plt.subplot(233)
#         sub.imshow(i1, cmap='gray')
#         i1 = image[nnn, :, :, 3]
#         sub = plt.subplot(234)
#         sub.imshow(i1, cmap='gray')
#         i1 = label[nnn, :, :]
#         sub = plt.subplot(235)
#         sub.imshow(i1, cmap='gray')
#         i1 = label[nnn, :, :]
#         sub = plt.subplot(236)
#         sub.imshow(i1, cmap='gray')
#         plt.show()