from Homework1.part1 import linalg
from Homework1.part1 import imageManip

import os
import numpy as np
from skimage import io
import utils
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 数据集根路径
data_path = r'E:\大三下科目\计算机视觉\实验\实验1\Homework1\part2\data'
# 读入数据集
images_divided_by_people = utils.collect_images(data_path)
# for index in range(len(images_divided_by_people)):
#     print("current index: {} images num: {}".format(index, len(images_divided_by_people[index])))
# print(len(images_divided_by_people[0]))

# 划分训练集和测试集
training_rate = 0.8
train_set = []
test_set = []
for people in range(len(images_divided_by_people)):
    cur_people_images = images_divided_by_people[people]
    train_length = int(len(cur_people_images) * training_rate)
    train_set.append(cur_people_images[:train_length])
    test_set.append(cur_people_images[train_length:])

train_set = np.array(train_set)
test_set = np.array(test_set)

# 将训练集和测试集进行降维
# 因为按照不同的人像进行划分
train_people_index, train_img_num, train_img_size = train_set.shape
test_people_index, test_img_num, test_img_size = test_set.shape
# print(people_index, img_num, img_size)
# train_set = train_set.reshape(people_index * img_num, img_size)
# print(train_set)
train_set = utils.covert_3d_to_2d(train_set)
test_set = utils.covert_3d_to_2d(test_set)
print("train data set dimensions: ", train_set.shape)
print("test data set dimensions: ", test_set.shape)

# 使用PCA降维并输出训练集、测试集维度和得到的特征向量维度
pca = PCA(n_components=100)
train_set_compressed = pca.fit_transform(train_set)
test_set_compressed = pca.transform(test_set)
print("train set compressed shape: ", train_set_compressed.shape)
print("test set compressed shape: ", test_set_compressed.shape)

# 从压缩后的特征空间进行逆变换
reconstructed_train_set = pca.inverse_transform(train_set_compressed)
reconstructed_test_set = pca.inverse_transform(test_set_compressed)

# 将重建的特征脸恢复成三维矩阵形式
reconstructed_train_set = reconstructed_train_set.reshape(train_people_index * train_img_num, train_img_size)
reconstructed_test_set = reconstructed_test_set.reshape(test_people_index * test_img_num, test_img_size)
print("reconstructed train set shape: ", reconstructed_train_set.shape)
print("reconstructed test set shape: ", reconstructed_test_set.shape)

# 重建特征脸（训练集）五张随机照片
num_samples = 5
train_title = "Comparison of Original and Reconstructed image from train data set"
test_title = "Comparison of Original and Reconstructed image from test data set"
utils.compare_original_and_reconstructed_image(train_set, reconstructed_train_set, num_samples, train_title)
utils.compare_original_and_reconstructed_image(test_set, reconstructed_test_set, num_samples, test_title)

# 输出降维后每个新特征向量所占的信息量百分比
explained_variance_ratio = pca.explained_variance_ratio_
for i, ratio in enumerate(explained_variance_ratio):
    print(f"Component {i + 1}: {ratio * 100:.2f}% of total variance")

# 计算所有返回特征所携带的信息量总和
total_variance = np.sum(explained_variance_ratio)
print(f"Total variance explained by all components: {total_variance * 100:.2f}% of total variance in the data")
print(f"Ratio of total variance explained: {total_variance * 100:.2f}%")

# 设置特征个数上限
max_features = 150

# 计算每个特征个数对应的信息量总和
total_variance_list = []
for n_components in range(1, max_features + 1):
    pca = PCA(n_components=n_components)
    pca.fit(reconstructed_train_set)
    total_variance = np.sum(pca.explained_variance_ratio_)
    total_variance_list.append(total_variance)

# 画出特征个数和所携带信息数的曲线图
plt.plot(range(1, max_features+1), total_variance_list)
plt.xlabel('Number of Features')
plt.ylabel('Total Variance Explained')
plt.title('Total Variance Explained by Number of Features')
plt.show()

# 训练标签
train_labels = [num for num in range(40) for _ in range(8)]
test_labels = [num for num in range(40) for _ in range(2)]

# 初始化特征保留数和准确率的列表
n_features_list = []
accuracy_list = []

# 逐个保留特征并训练KNN分类器
for n_features in range(1, max_features+1):
    # 保留前n个特征
    pca = PCA(n_components=n_features)
    reduced_train_data = pca.fit_transform(train_set)
    reduced_test_data = pca.transform(test_set)

    # 训练KNN分类器
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(reduced_train_data, train_labels)

    # 在测试集上进行预测并计算准确率
    predictions = knn.predict(reduced_test_data)
    accuracy = accuracy_score(test_labels, predictions)

    # 添加特征保留数和准确率到列表
    n_features_list.append(n_features)
    accuracy_list.append(accuracy)

# 画出特征保留数和准确率的曲线图
plt.plot(n_features_list, accuracy_list)
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.title('Accuracy by Number of Features')
plt.show()

