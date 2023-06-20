# Part 1 理论推导
> 200110625 柯炽炜

## 假设
- 特征图的尺寸为 $3 * 3$
- 卷积核和池化核的尺寸为 $2 * 2$
- 无填充
- 步长为1

## 卷积层前向传播
问题描述：给定输入特征图$X$, 卷积层输出特征图$Y$，卷积核$W$和偏置$b$，请推导卷积核的前向传播计算过程

题解：

- 输入特征图 $X$ 的深度为 $D$，记作 $X \in \mathbb{R}^{3 \times 3 \times D}$
- 卷积核 $W$ 的深度与输入特征图相同，记作 $W \in \mathbb{R}^{2 \times 2 \times D}$
- 输出特征图 $Y$ 的深度也为 $D$，记作 $Y \in \mathbb{R}^{2 \times 2 \times D}$

对于输出特征图 $Y$ 的每一个元素 $Y(i, j, k)$，其中 $i$ 和 $j$ 是特征图上的空间位置，$k$ 是深度维度上的索引，我们可以按照以下公式进行计算：

$$
Y(i, j, k) = \sum_{m=0}^{1} \sum_{n=0}^{1} \sum_{l=0}^{D-1} X(i+m, j+n, l) \cdot W(m, n, l, k) + b(k)
$$
其中，$X(i+m, j+n, l)$ 表示输入特征图 $X$ 在位置 $(i+m, j+n)$ 处，深度索引为 $l$ 的元素；$W(m, n, l, k)$ 表示卷积核 $W$ 在位置 $(m, n)$ 处，深度索引为 $l$ 的元素；$b(k)$ 表示偏置项。

根据卷积操作的定义，我们可以看到，输出特征图 $Y$ 的每个元素都是通过对应位置上的输入特征图和卷积核的元素逐元素相乘，并对结果求和得到。这样，我们就完成了卷积层的前向传播计算过程。

## 卷积层反向传播

问题描述：给定输入特征图$X$, 卷积层输出特征图$Y$，损失函数$L$，卷积层的梯度输出$\frac{\partial L}{\partial Y}$，请推导计算输入梯度$\frac{\partial L}{\partial X}$、权重梯度$\frac{\partial L}{\partial W}$ 和 偏置梯度 $\frac{\partial L}{\partial b}$ 的过程

题解：

首先，我们来推导输入梯度 $\frac{\partial L}{\partial X}$ 的计算过程。

根据链式法则，我们有：

$$
\frac{\partial L}{\partial X(i, j, l)} = \sum_{m=0}^{1} \sum_{n=0}^{1} \sum_{k=0}^{D-1} \frac{\partial L}{\partial Y(m, n, k)} \cdot \frac{\partial Y(m, n, k)}{\partial X(i, j, l)}
$$
对于 $\frac{\partial Y(m, n, k)}{\partial X(i, j, l)}$，我们可以观察到只有在位置 $(i-m, j-n)$ 处存在非零的元素，对应于卷积核的位置 $(m, n)$：

$$
\frac{\partial Y(m, n, k)}{\partial X(i, j, l)} = \begin{cases}
W(m, n, l, k) & \text{if } i-m \text{ and } j-n \text{ are valid indices} \\
0 & \text{otherwise}
\end{cases}
$$
将其代入链式法则的公式中，我们可以得到输入梯度的计算公式：

$$
\frac{\partial L}{\partial X(i, j, l)} = \sum_{m=0}^{1} \sum_{n=0}^{1} \sum_{k=0}^{D-1} \frac{\partial L}{\partial Y(m, n, k)} \cdot W(m, n, l, k)
$$
上述公式表达了损失函数对输入特征图 $X$ 的梯度。通过计算输入梯度，我们可以了解损失函数对输入特征图中每个元素的贡献程度。

接下来，我们将推导权重梯度 $\frac{\partial L}{\partial W}$ 的计算过程。

根据链式法则，我们有：

$$
\frac{\partial L}{\partial W(m, n, l, k)} = \sum_{i=0}^{1} \sum_{j=0}^{1} \sum_{k'=0}^{D-1} \frac{\partial L}{\partial Y(i, j, k')} \cdot \frac{\partial Y(i, j, k')}{\partial W(m, n, l, k)}
$$
对于 $\frac{\partial Y(i, j, k')}{\partial W(m, n, l, k)}$，我们可以观察到只有在位置 $(i-m, j-n)$ 处存在非零的元素，对应于输入特征图的位置 $(i, j)$：

$$
\frac{\partial Y(i, j, k')}{\partial W(m, n, l, k)} = \begin{cases}
X(i-m, j-n, l) & \text{if } i-m \text{ and } j-n \text{ are valid indices} \\
0 & \text{otherwise}
\end{cases}
$$
将其代入链式法则的公式中，我们可以得到权重梯度的计算公式：

$$
\frac{\partial L}{\partial W(m, n, l, k)} = \sum_{i=0}^{1} \sum_{j=0}^{1} \sum_{k'=0}^{D-1} \frac{\partial L}{\partial Y(i, j, k')} \cdot X(i-m, j-n, l)
$$
上述公式表达了损失函数对卷积核 $W$ 的梯度。通过计算权重梯度，我们可以了解损失函数对卷积核中每个元素的贡献程度。

最后，我们将推导偏置梯度 $\frac{\partial L}{\partial b}$ 的计算过程。

偏置 $b$ 是一个标量，因此偏置梯度 $\frac{\partial L}{\partial b}$ 等于损失函数对输出特征图 $Y$ 的梯度之和：

$$
\frac{\partial L}{\partial b} = \sum_{i=0}^{1} \sum_{j=0}^{1} \sum_{k=0}^{D-1} \frac{\partial L}{\partial Y(i, j, k)}
$$
上述公式表达了损失函数对偏置项 $b$ 的梯度。通过计算偏置梯度，我们可以了解损失函数对偏置的贡献程度。

通过推导上述公式，我们可以计算输入梯度 $\frac{\partial L}{\partial X}$、权重梯度 $\frac{\partial L}{\partial W}$ 和偏置梯度 $\frac{\partial L}{\partial b}$，从而实现卷积层的反向传播过程。

## 池化层前向传播

问题描述：给定输入特征图$X$，池化层输出特征图$Y$，请推导最大池化层的前向传播计算过程

题解：

- 输入特征图 $X$ 的深度为 $D$，记作 $X \in \mathbb{R}^{3 \times 3 \times D}$。
- 输出特征图 $Y$ 的深度与输入特征图相同，记作 $Y \in \mathbb{R}^{2 \times 2 \times D}$。

对于输出特征图 $Y$ 的每一个元素 $Y(i, j, k)$，其中 $i$ 和 $j$ 是特征图上的空间位置，$k$ 是深度维度上的索引，我们可以表示为：

$$
Y(i, j, k) = \max(X(i\times2:i\times2+1, j\times2:j\times2+1, k))
$$
其中，$\max(X(i\times2:i\times2+1, j\times2:j\times2+1, k))$ 表示： 
$$
在局部区域X(i\times2:i\times2+1, j\times2:j\times2+1, k)中找到最大值。
$$
这里的局部区域 $X(i\times2:i\times2+1, j\times2:j\times2+1, k)$ 表示:

- 输入特征图 $X$ 中以位置 $(i, j)$ 为中心的 $2 \times 2$ 区域，对应于输出特征图的位置 $(i, j, k)$。
- 在这个局部区域中，我们找到最大的元素，然后将其赋值给输出特征图 $Y(i, j, k)$。

通过以上数学表达式，我们可以得到最大池化层的前向传播计算过程。

## 池化层反向传播

问题描述：给定输入特征图$X$, 卷积层输出特征图$Y$，损失函数$L$，最大池化层的梯度输出$\frac{\partial L}{\partial Y}$，请推导计算输入梯度$\frac{\partial L}{\partial X}$的过程

题解：

最大池化层没有可训练的参数，因此我们只需要计算 $\frac{\partial L}{\partial X}$，即损失函数对输入特征图 $X$ 的梯度。

在最大池化层的前向传播中，我们通过在局部区域中选择最大值来计算输出特征图 $Y$。

因此，在计算输入梯度时，我们需要确定损失函数对每个局部区域的最大值的位置。对于这些位置，我们将梯度 $\frac{\partial L}{\partial Y}$ 回传给对应的位置。

具体而言，对于输入特征图 $X$ 的每个元素 $X(i, j, k)$，我们有：

$$
\frac{\partial L}{\partial X(i, j, k)} = \sum_{p=0}^{1} \sum_{q=0}^{1} \frac{\partial L}{\partial Y\left(\left\lfloor\frac{i}{2}\right\rfloor, \left\lfloor\frac{j}{2}\right\rfloor, k\right)} \cdot \frac{\partial Y\left(\left\lfloor\frac{i}{2}\right\rfloor, \left\lfloor\frac{j}{2}\right\rfloor, k\right)}{\partial X(i, j, k)}
$$
其中，$\left\lfloor\frac{i}{2}\right\rfloor$ 和 $\left\lfloor\frac{j}{2}\right\rfloor$ 表示整数除法，即向下取整。

对于 $\frac{\partial Y\left(\left\lfloor\frac{i}{2}\right\rfloor, \left\lfloor\frac{j}{2}\right\rfloor, k\right)}{\partial X(i, j, k)}$，我们可以观察到只有在位置 $(i', j')$ 处存在非零的元素，其中 $i'$ 和 $j'$ 是通过局部区域映射而来的位置：

$$
\frac{\partial Y\left(\left\lfloor\frac{i}{2}\right\rfloor, \left\lfloor\frac{j}{2}\right\rfloor, k\right)}{\partial X(i, j, k)} = \begin{cases}
1 & \text{if } i = 2i' \text{ and } j = 2j' \\
0 & \text{otherwise}
\end{cases}
$$
将上述结果代入到计算输入梯度的公式中，我们可以得到：

$$
\frac{\partial L}{\partial X(i, j, k)} = \frac{\partial L}{\partial Y\left(\left\lfloor\frac{i}{2}\right\rfloor, \left\lfloor\frac{j}{2}\right\rfloor, k\right)} \cdot \frac{\partial Y\left

(\left\lfloor\frac{i}{2}\right\rfloor, \left\lfloor\frac{j}{2}\right\rfloor, k\right)}{\partial X(i, j, k)}
$$
综上所述，我们可以通过上述公式计算输入梯度 $\frac{\partial L}{\partial X}$，从而实现最大池化层的反向传播过程。



