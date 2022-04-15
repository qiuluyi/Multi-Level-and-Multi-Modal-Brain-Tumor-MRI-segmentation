import cv2 as cv
from Fuzzy.readMHA_something import BRATS2015 as Brats
import matplotlib.pyplot as plt
import scipy as scp
import numpy as np
brats = Brats(8)

for i in range(16+31*11):
    image, label = brats.next_train_batch(5)

image_f = image[:, :, :, 0]
image_t1 = image[:, :, :, 1]
image_t1c = image[:, :, :, 2]
image_t2 = image[:, :, :, 3]
# kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
# dilation = np.asarray(image[:, :, :, 2].copy(), dtype=np.uint8)
# for j in range(image.shape[0]):
#     dilation[j, :, :] = cv.dilate(dilation[j, :, :], kernel)
img_temp_1 = image_t1c.copy()
img_temp_2 = image_t1c.copy()
img_temp_2_2 = image_t1c.copy()
img_temp_3 = image_t1c.copy()
img_temp_4 = image_t1c.copy()

# dilation = np.expand_dims(dilation, axis=-1)
# img_temp = np.where(dilation > 0, 0, img_temp)
for k in range(img_temp_1.shape[0]):
    img_temp_1[k, :, :] = cv.GaussianBlur(img_temp_1[k, :, :], (0, 0), 1)
for k in range(img_temp_1.shape[0]):
    img_temp_2[k, :, :] = cv.GaussianBlur(img_temp_2[k, :, :], (0, 0), 3)
for k in range(img_temp_1.shape[0]):
    img_temp_2_2[k, :, :] = cv.GaussianBlur(img_temp_2_2[k, :, :], (25, 25), 3)
for k in range(img_temp_2.shape[0]):
    img_temp_3[k, :, :] = cv.GaussianBlur(img_temp_3[k, :, :], (0, 0), 5)
for k in range(img_temp_3.shape[0]):
    img_temp_4[k, :, :] = cv.GaussianBlur(img_temp_4[k, :, :], (0, 0), 7)
# img_gaussian = [cv.GaussianBlur(x, (3, 3), 20) for x in img_temp]
# img_gaussian = np.concatenate(img_gaussian, axis=0)
test = img_temp_2_2 - img_temp_2

for modal in range(len(img_org[0, 0, 0, :]) - 1):
    img_multimodal = np.concatenate((img_multimodal, np.concatenate((img_org[0, :, :, modal + 1], img[0, :, :, modal + 1]), axis=1)), axis=0)
scp.misc.imsave(brats.mhaName + 'fuzzy_img.png', img_multimodal)


print(np.max(test))
print(np.min(test))
for nnn in range(4):
    if np.max(label)>0:
        i1 = image[nnn, :, :, 0]
        sub = plt.subplot(341)
        sub.imshow(i1, cmap='gray')
        i1 = image[nnn, :, :, 1]
        sub = plt.subplot(342)
        sub.imshow(i1, cmap='gray')
        i1 = image[nnn, :, :, 2]
        sub = plt.subplot(343)
        sub.imshow(i1, cmap='gray')
        i1 = image[nnn, :, :, 3]
        sub = plt.subplot(344)
        sub.imshow(i1, cmap='gray')
        i1 = label[nnn, :, :]
        sub = plt.subplot(345)
        sub.imshow(i1, cmap='gray')
        i1 = img_temp_1[nnn, :, :]
        sub = plt.subplot(346)
        sub.imshow(i1, cmap='gray')
        i2 = img_temp_2[nnn, :, :]
        sub = plt.subplot(347)
        sub.imshow(i2, cmap='gray')
        i2 = img_temp_2_2[nnn, :, :]
        sub = plt.subplot(348)
        sub.imshow(i2, cmap='gray')
        i2 = img_temp_3[nnn, :, :]
        sub = plt.subplot(349)
        sub.imshow(i2, cmap='gray')
        i2 = img_temp_4[nnn, :, :]
        sub = plt.subplot(3,4,10)
        sub.imshow(i2, cmap='gray')


        plt.show()

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