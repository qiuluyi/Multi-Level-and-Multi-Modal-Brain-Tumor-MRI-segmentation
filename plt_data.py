import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
class plt_save():
       def save_feature_map(path, data):
              plt.figure()
              fig = plt.gcf()
              plt.axis('off')
              # cmap="viridis"：0.8 黄到蓝 -0.8
              plt.imshow(data, cmap='viridis')
              #plt.imshow(data, cmap='hot')
              # 去除图像周围的白边
              height, width = data.shape
              # 如果dpi=300，那么图像大小=height*width
              fig.set_size_inches(width / 100.0, height / 100.0)
              plt.gca().xaxis.set_major_locator(plt.NullLocator())
              plt.gca().yaxis.set_major_locator(plt.NullLocator())
              plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
              plt.margins(0, 0)
              plt.savefig(path, dpi=300)

       def dice_line(self):
              plt.rcParams['font.sans-serif'] = ['SimHei']  # 处理中文无法正常显示的问题 成功
              plt.rcParams['axes.unicode_minus'] = False  # 负号显示
              # plt.xlabel("这是x轴")  # 设置x轴名称
              # plt.ylabel("这是y轴")  # 设置y轴名称
              # plt.title("这是标题")  # 设置标题
              x = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 虚假的x值，用来等间距分割
              x_index = ['0', '1', '3', '5', '7', '9', '11', '13', '15']  # x 轴显示的刻度
              list1 = [0, 0.790861338417494, 0.825832257364377, 0.829535822349319, 0.743953843452811, 0.887163705831125,
                       0.897386410937020, 0.895249780365137, 0.874798670545903]  # y值
              list2 = [0, 0.826822966077835,
                       0.848151947240295,
                       0.833398514024436,
                       0.785504655182942,
                       0.892393835096190,
                       0.901412741738544,
                       0.897634807173072,
                       0.876734447208229]
              pow_num = 1
              list1 = np.power(list1, pow_num)
              list2 = np.power(list2, pow_num)
              y = np.power([0, 0.5, 0.6, 0.7, 0.8, 0.9, 1], pow_num)
              y = np.append(y, [0.81, 1])
              y_index = ['0', '0.5', '0.6', '0.7', '0.8', '0.9', '1']
              plt.yticks(y, y_index)

              # power_smooth = spline(T, power, xnew)
              plt.plot(x, list1, marker='d')
              plt.plot(x, list2, marker='x')
              _ = plt.xticks(x, x_index)  # 显示坐标字
              plt.show()


path = 'RESNEXT_unet_20210105_02c_4' + 'X/'
saveImgPath = 'image_result/Image_train_' + path

featurePath = saveImgPath + 'feature/'
down_1 = '_55906' + '_feature_down_1.png'
down_4 = '_55906' + '_feature_down_4.png'
up_4 = '_55906' + '_feature_up_4.png'

down_ptl_1 = '_55906' + '_feature_down_1_plt.png'
down_plt_4 = '_55906' + '_feature_down_4_plt.png'
up_plt_4 = '_55906' + '_feature_up_4_plt.png'

down_1_im = mpimg.imread(featurePath + down_1)
plt_save.save_feature_map(featurePath + down_ptl_1, down_1_im)

down_4_im = mpimg.imread(featurePath + down_4)
plt_save.save_feature_map(featurePath + down_plt_4, down_4_im)

up_4_im = mpimg.imread(featurePath + up_4)
plt_save.save_feature_map(featurePath + up_plt_4, up_4_im)
