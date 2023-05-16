import os
import numpy as np
import skimage.io
from Homework1.part1 import linalg
from Homework1.part1 import imageManip
import matplotlib.pyplot as plt

def collect_images(root_path):
    """
    批量读入根目录下所有图片
    :param root_path: 数据集根目录
    :return:
    """
    images = []
    # 对目录下的文件进行遍历
    for file in os.listdir(root_path):
        # 对于是文件的情况
        if os.path.isfile(os.path.join(root_path, file)):
            c = os.path.basename(file)
            name = root_path + '\\' + c
            img = skimage.io.imread(name, as_gray=True)
            # 需要对图像进行扁平化处理
            flattened_img = img.flatten()
            # img = skimage.transform.resize(img, (112, 92))
            images.append(flattened_img)
        # 对于是文件夹的情况，递归调用
        else:
            sub_images = collect_images(os.path.join(root_path, file))
            images.append(sub_images)
    return images


def cal_eigenvalues_and_eigenvectors(data_set, dimension):
    """
    计算数据集的特征值和特征向量
    :param data_set: 人脸数据集
    :param dimension: 维度，即特征值和特征向量个数
    :return: 含有dimension个特征值和特征向量的list
    """
    eig = []
    for people in range(len(data_set)):
        cur_people_images = data_set[people]
        cur_people_eig = []
        for img in cur_people_images:
            # 计算特征值和特征向量, 取前100个
            eigenvalues, eigenvectors = linalg.get_eigen_values_and_vectors(img, dimension)
            cur_people_eig.append([eigenvalues, eigenvectors])
        eig.append(cur_people_eig)
    return eig


def covert_3d_to_2d(data_set):
    """
    将三维的数据结构转化为2维的
    :param data_set: 原始数据集
    :return: 转化后的数据集
    """
    people_index, img_num, img_size = data_set.shape
    return data_set.reshape(people_index * img_num, img_size)


def compare_original_and_reconstructed_image(original_data_set, reconstructed_data_set, img_num, title):
    """
    重建数据集
    :param original_data_set: 原始数据集
    :param reconstructed_data_set: 重建后的数据集
    :param img_num: 需要展示的图片数量
    :param title: 图片标题
    :return:
    """
    fig, axes = plt.subplots(img_num, 2, figsize=(10, 10))
    for index in range(img_num):
        original_img = original_data_set[index].reshape(112, 92)
        reconstructed_img = reconstructed_data_set[index].reshape(112, 92)

        # 原始图像
        axes[index, 0].imshow(original_img, cmap='gray')
        axes[index, 0].set_title("Original")

        # 恢复图像
        axes[index, 1].imshow(reconstructed_img, cmap='gray')
        axes[index, 1].set_title("Reconstructed")
        # io.imshow(original_img)
        # io.show()
        # io.imshow(reconstructed_img)
        # io.show()
    # 调整子图间距
    plt.tight_layout()
    # 调整间距
    fig.subplots_adjust(top=0.9)
    # 添加总标题
    fig.suptitle(title)
    # 显示图像
    plt.show()

