import numpy as np
import random
import math
from scipy.spatial.distance import squareform, pdist, cdist
from skimage.util import img_as_float
from skimage import color


# Clustering Methods for 1-D points
def kmeans(features, k, num_iters=500):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """
    # 随机初始化簇中心点
    cluster_centers = features[np.random.choice(len(features), k)]
    for i in range(num_iters):
        # 计算每个点与簇中心点的距离
        distances = np.linalg.norm(features[:, None] - cluster_centers, axis=2)
        # 将点分配到最近的簇
        assignments = np.argmin(distances, axis=1)
        # 计算新的簇中心点
        new_cluster_centers = np.array([features[assignments == j].mean(axis=0) for j in range(k)])
        # 判断是否满足终止条件，如果满足则停止迭代
        if np.allclose(cluster_centers, new_cluster_centers):
            break
        cluster_centers = new_cluster_centers
    # 返回每个点的簇分配结果
    return assignments


# Clustering Methods for colorful image
def kmeans_color(features, k, num_iters=500):
    # 保存原始特征形状
    original_shape = features.shape
    # 将特征展平成二维数组
    features = features.reshape(-1, features.shape[-1])
    # 随机初始化簇中心点
    centers = features[np.random.choice(len(features), k)]

    i = 0
    while i < num_iters:
        # 计算每个点与簇中心点的距离
        distances = np.linalg.norm(features[:, None] - centers, axis=2)
        # 将点分配到最近的簇
        assignments = np.argmin(distances, axis=1)
        # 计算新的簇中心点
        new_centers = np.array(
            [features[assignments == j].mean(axis=0) for j in range(k)])
        # 判断是否满足终止条件，如果满足则停止迭代
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
        i += 1

    # 将簇分配结果恢复成原始特征形状
    assignments = assignments.reshape(original_shape[:2])
    # 返回簇分配结果
    return assignments


# 找每个点最后会收敛到的地方（peak）
def findpeak(data, idx, r):
    # 收敛阈值
    t = 0.01
    # 移动距离
    shift = np.array([1])
    # 获取指定索引的数据点
    data_point = data[:, idx]
    # 转置数据，便于计算距离
    dataT = data.T
    # 转置数据点，便于计算距离
    data_pointT = data_point.T
    data_pointT = data_pointT.reshape(1, 3)

    while (shift > t).any():
        # 计算当前点与所有点之间的距离
        dist = np.linalg.norm(dataT - data_point, axis=1)
        # 筛选在半径 r 内的点
        data_within_r = dataT[dist < r]
        # 计算半径内点的均值向量作为新的点位置
        data_point_new = np.mean(data_within_r, axis=0)
        # 计算点的移动距离
        shift = data_point_new - data_point
        # 更新当前点的位置为新的点位置
        data_point = data_point_new

    return data_point


# Mean shift algorithm
# 可以改写代码，鼓励自己的想法，但请保证输入输出与notebook一致
def meanshift(data, r):
    labels = np.zeros(len(data.T))  # 存储数据点的类别标签
    peaks = []  # 聚类的中心点
    label_no = 1  # 当前标签
    labels[0] = label_no

    # 针对第一个索引调用find-peak函数
    peak = findpeak(data, 0, r)
    peaks.append(peak)

    # 遍历每个数据点
    for idx in range(1, len(data.T)):
        # 寻找当前点的聚类中心（peak）
        peak = findpeak(data, idx, r)
        # 实时检查当前聚类中心是否收敛到一个已有聚类中心（与已有的peaks比较）
        dist = np.linalg.norm(np.array(peaks) - peak, axis=1)
        dist_min = np.min(dist)
        if dist_min < r:
            # 若不收敛，将当前点归属到最近的聚类中心
            labels[idx] = np.argmin(dist) + 1
        else:
            # 若收敛，更新标签号、聚类中心、数据点的标签，并继续遍历
            label_no += 1
            labels[idx] = label_no
            peaks.append(peak)

    return labels, np.array(peaks).T


# image segmentation
def segmIm(img, r):
    # Image gets reshaped to a 2D array
    img_reshaped = np.reshape(img, (img.shape[0] * img.shape[1], 3))

    # We will work now with CIELAB images
    imglab = color.rgb2lab(img_reshaped)
    # segmented_image is declared
    segmented_image = np.zeros((img_reshaped.shape[0], img_reshaped.shape[1]))

    labels, peaks = meanshift(imglab.T, r)
    # Labels are reshaped to only one column for easier handling
    labels_reshaped = np.reshape(labels, (labels.shape[0], 1))

    # We iterate through every possible peak and its corresponding label
    for label in range(0, peaks.shape[1]):
        # Obtain indices for the current label in labels array
        inds = np.where(labels_reshaped == label + 1)[0]

        # The segmented image gets indexed peaks for the corresponding label
        corresponding_peak = peaks[:, label]
        segmented_image[inds, :] = corresponding_peak
    # The segmented image gets reshaped and turn back into RGB for display
    segmented_image = np.reshape(segmented_image, (img.shape[0], img.shape[1], 3))

    res_img = color.lab2rgb(segmented_image)
    res_img = color.rgb2gray(res_img)
    return res_img


# Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = None

    # 通过比较 mask_gt 和 mask 数组的相等性来计算准确率
    # accuracy = np.sum(mask_gt == mask) / mask_gt.size

    # 计算两种可能的错误情况的准确率
    accuracy1 = np.sum(np.abs(mask_gt - mask)) / mask_gt.size
    accuracy2 = np.sum(np.abs(mask_gt - (1 - mask))) / mask_gt.size

    # 取两种错误情况的准确率中的最大值作为最终的准确率
    accuracy = max(accuracy1, accuracy2)

    return accuracy
