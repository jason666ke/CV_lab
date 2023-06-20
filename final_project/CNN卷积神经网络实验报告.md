# 使用Pytorch框架实现CNN图像分类任务
> 200110625 柯炽炜

## 模型结构
- 卷积层1：输入通道为3，输出通道为16，卷积核大小为3x3，padding为1，采用ReLU激活函数。
- 卷积层2：输入通道为16，输出通道为32，卷积核大小为3x3，padding为1，采用ReLU激活函数。
- 卷积层3：输入通道为32，输出通道为64，卷积核大小为3x3，padding为1，采用ReLU激活函数。
- 最大池化层：核大小为2x2，步幅为2。
- 全连接层1：输入节点数为64x4x4，输出节点数为500，采用ReLU激活函数。
- 全连接层2：输入节点数为500，输出节点数为10。

## 超参数选择
- 批量大小（batch_size）：16
- 学习率（lr）：0.01
- 训练迭代次数（n_epochs）：30
- 损失函数：交叉熵损失函数（CrossEntropyLoss）
- 优化器：随机梯度下降（SGD）

## 训练过程
1. 在训练过程中，每个epoch都会对训练集和验证集进行一次迭代。
2. 训练集模型：对于每个训练批次，将输入数据传递给模型，计算输出，并计算损失。然后进行反向传播和梯度更新。
3. 验证集模型：对于每个验证批次，将输入数据传递给模型，计算输出并计算损失，以评估模型在验证集上的性能。
4. 保存最小验证损失的模型参数。
5. 打印每个epoch的训练损失和验证损失。

## 模型评估方法
- 加载最小验证损失的模型参数。
- 对测试集进行预测并计算测试损失。
- 计算每个类别的测试准确率和总体准确率。

## 性能测试
- 训练集测试结果：
    ~~~
    Test in test data set
    Test Loss: 0.243731
    
    Test Accuracy of airPlane: 94% (3781/4002)
    Test Accuracy of autoMobile: 96% (3844/3999)
    Test Accuracy of  bird: 86% (3422/3975)
    Test Accuracy of   cat: 84% (3402/4039)
    Test Accuracy of  deer: 90% (3569/3959)
    Test Accuracy of   dog: 83% (3382/4028)
    Test Accuracy of  frog: 96% (3893/4050)
    Test Accuracy of horse: 93% (3734/4011)
    Test Accuracy of  ship: 96% (3865/4012)
    Test Accuracy of truck: 94% (3708/3925)
    
    Test Accuracy (Overall): 91% (36600/40000)
    ~~~
- 验证集测试结果
    ~~~
    Test in validation set
    Test Loss: 0.243731
    
    Test Accuracy of airPlane: 94% (3781/4002)
    Test Accuracy of autoMobile: 96% (3844/3999)
    Test Accuracy of  bird: 86% (3422/3975)
    Test Accuracy of   cat: 84% (3402/4039)
    Test Accuracy of  deer: 90% (3569/3959)
    Test Accuracy of   dog: 83% (3382/4028)
    Test Accuracy of  frog: 96% (3893/4050)
    Test Accuracy of horse: 93% (3734/4011)
    Test Accuracy of  ship: 96% (3865/4012)
    Test Accuracy of truck: 94% (3708/3925)
    
    Test Accuracy (Overall): 91% (36600/40000)
    ~~~
- 测试集测试结果
    ~~~
    Test Loss: 0.725465
    
    Test Accuracy of airPlane: 82% (821/1000)
    Test Accuracy of autoMobile: 86% (863/1000)
    Test Accuracy of  bird: 65% (652/1000)
    Test Accuracy of   cat: 56% (561/1000)
    Test Accuracy of  deer: 69% (694/1000)
    Test Accuracy of   dog: 63% (636/1000)
    Test Accuracy of  frog: 87% (873/1000)
    Test Accuracy of horse: 80% (800/1000)
    Test Accuracy of  ship: 87% (872/1000)
    Test Accuracy of truck: 80% (807/1000)
    
    Test Accuracy (Overall): 75% (7579/10000)
    ~~~

## 模型分析
### 训练集和验证集的性能
- 在训练集上，模型表现良好，准确率高。具体而言，飞机（airPlane）和轮船（ship）类别的准确率达到了94%和96%，其他类别也有较高的准确率。
- 在验证集上，模型的准确率与训练集表现相似。准确率最高的类别是青蛙（frog）和轮船（ship），分别达到了96%的准确率。

### 测试集的性能
- 在测试集上，模型的准确率相对较低，整体准确率为75%。准确率最高的类别是青蛙（frog）和轮船（ship），分别达到了87%的准确率。
- 猫（cat）和狗（dog）类别的准确率相对较低，分别为56%和63%。

### 总体分析
- 模型在训练集和验证集上表现良好，具有较高的准确率。这表明模型能够很好地学习训练数据中的模式和特征，并且在未见过的数据上能够进行良好的预测。
- 然而，在测试集上，模型的性能略有下降，准确率较训练集和验证集有所降低。这可能是由于测试集中的样本具有更多的变化和噪声，使得模型难以准确地进行分类。
- 特别是在猫（cat）和狗（dog）类别上，模型的准确率较低，可能需要进一步调整模型结构、优化超参数或增加更多的训练数据来改进这些类别的分类能力。

综上所述，模型在整体上具有良好的性能，但仍有改进的空间。进一步的实验和调整可以帮助提高模型在测试集上的准确率，尤其是对于低准确率的类别。

