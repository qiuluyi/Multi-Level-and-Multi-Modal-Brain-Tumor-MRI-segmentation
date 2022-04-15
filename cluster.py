import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from Fuzzy.readMHA_something import BRATS2015 as Brats
brats = Brats(8)

from itertools import chain

def seg_kmeans_color():
    for i in range(15):
        image, label = brats.next_train_batch(5)
    # img = cv2.imread('000129.jpg', cv2.IMREAD_COLOR)
    # # 变换一下图像通道bgr->rgb，否则很别扭啊
    # b, g, r = cv2.split(img)
    # img = cv2.merge([r, g, b])
    img = image[0,:,:,:]
    label_min = label[0, :, :]

    # 3个通道展平
    img_flat = img.reshape((img.shape[0] * img.shape[1], 4))
    img_flat = np.float32(img_flat)

    # 迭代参数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 20, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # 聚类
    compactness, labels, centers = cv2.kmeans(img_flat, 5, None, criteria, 10, flags)

    # 显示结果
    img_output = labels.reshape((img.shape[0], img.shape[1]))
    plt.subplot(131), plt.imshow(img), plt.title('input')
    plt.subplot(132), plt.imshow(img_output, 'gray'), plt.title('kmeans')
    plt.subplot(133), plt.imshow(label_min, 'gray'), plt.title('LABEL')
    plt.show()

#均值和方差
def mean_var():
    image_flair_1 = []
    image_t1_1 = []
    image_t1c_1 = []
    image_t2_1 = []

    image_flair_2 = []
    image_t1_2 = []
    image_t1c_2 = []
    image_t2_2 = []

    image_flair_3 = []
    image_t1_3 = []
    image_t1c_3 = []
    image_t2_3 = []

    image_flair_4 = []
    image_t1_4 = []
    image_t1c_4 = []
    image_t2_4 = []

    for i in tqdm(range((brats.train_batch * 155)),
                  total=((brats.train_batch * 155)),
                  ncols=70, leave=False, unit='b'):
        image_temp, label_temp = brats.next_train_batch(1)
        i_f = image_temp[0, :, :, 0]
        i_t1 = image_temp[0, :, :, 1]
        i_t1c = image_temp[0, :, :, 2]
        i_t2 = image_temp[0, :, :, 3]

        label = label_temp[0, :, :]
        class_num = 1
        image_flair_1.append(i_f[label == class_num])
        image_t1_1.append(i_t1[label == class_num])
        image_t1c_1.append(i_t1c[label == class_num])
        image_t2_1.append(i_t2[label == class_num])

        class_num = 2
        image_flair_2.append(i_f[label == class_num])
        image_t1_2.append(i_t1[label == class_num])
        image_t1c_2.append(i_t1c[label == class_num])
        image_t2_2.append(i_t2[label == class_num])

        class_num = 3
        image_flair_3.append(i_f[label == class_num])
        image_t1_3.append(i_t1[label == class_num])
        image_t1c_3.append(i_t1c[label == class_num])
        image_t2_3.append(i_t2[label == class_num])

        class_num = 4
        image_flair_4.append(i_f[label == class_num])
        image_t1_4.append(i_t1[label == class_num])
        image_t1c_4.append(i_t1c[label == class_num])
        image_t2_4.append(i_t2[label == class_num])
        # if i == 75:
        #     break
        pass
    # image_flair_total = reduce(operator.add, image_flair)
    image_flair_total_1 = list(chain(*image_flair_1))
    image_t1_total_1 = list(chain(*image_t1_1))
    image_t1c_total_1 = list(chain(*image_t1c_1))
    image_t2_total_1 = list(chain(*image_t2_1))

    image_flair_total_2 = list(chain(*image_flair_2))
    image_t1_total_2 = list(chain(*image_t1_2))
    image_t1c_total_2 = list(chain(*image_t1c_2))
    image_t2_total_2 = list(chain(*image_t2_2))

    image_flair_total_3 = list(chain(*image_flair_3))
    image_t1_total_3 = list(chain(*image_t1_3))
    image_t1c_total_3 = list(chain(*image_t1c_3))
    image_t2_total_3 = list(chain(*image_t2_3))

    image_flair_total_4 = list(chain(*image_flair_4))
    image_t1_total_4 = list(chain(*image_t1_4))
    image_t1c_total_4 = list(chain(*image_t1c_4))
    image_t2_total_4 = list(chain(*image_t2_4))

    print("class", 1)
    print("flair_mean_var", np.mean(image_flair_total_1), np.var(image_flair_total_1))
    print("t1_mean_var", np.mean(image_t1_total_1), np.var(image_t1_total_1))
    print("t1c_mean_var", np.mean(image_t1c_total_1), np.var(image_t1c_total_1))
    print("t2_mean_var", np.mean(image_t2_total_1), np.var(image_t2_total_1))
    
    print("class", 2)
    print("flair_mean_var", np.mean(image_flair_total_2), np.var(image_flair_total_2))
    print("t1_mean_var", np.mean(image_t1_total_2), np.var(image_t1_total_2))
    print("t1c_mean_var", np.mean(image_t1c_total_2), np.var(image_t1c_total_2))
    print("t2_mean_var", np.mean(image_t2_total_2), np.var(image_t2_total_2))
    
    print("class", 3)
    print("flair_mean_var", np.mean(image_flair_total_3), np.var(image_flair_total_3))
    print("t1_mean_var", np.mean(image_t1_total_3), np.var(image_t1_total_3))
    print("t1c_mean_var", np.mean(image_t1c_total_3), np.var(image_t1c_total_3))
    print("t2_mean_var", np.mean(image_t2_total_3), np.var(image_t2_total_3))
    
    print("class", 4)
    print("flair_mean_var", np.mean(image_flair_total_4), np.var(image_flair_total_4))
    print("t1_mean_var", np.mean(image_t1_total_4), np.var(image_t1_total_4))
    print("t1c_mean_var", np.mean(image_t1c_total_4), np.var(image_t1c_total_4))
    print("t2_mean_var", np.mean(image_t2_total_4), np.var(image_t2_total_4))

if __name__ == '__main__':
    #seg_kmeans_color()
    mean_var()
