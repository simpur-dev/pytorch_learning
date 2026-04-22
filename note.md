# PyTorch

```python
#导入PyTorch工具包
import torch  
#创建了一个张量，并告诉python，我要追踪这个变量的计算过程，帮我自动算导数
x = torch.tensor(3.0, requires_grad=True)
#数学公式
y = x ** 2 + 2 * x
#自动反向传播求导，并把结果存在x.grad里面
y.backward()
#x.grad:存储了自动算好的导数结果，item()可以把PyTorch的张量转换成普通数字
print(x.grad.item())
```

## 第 1 课：张量（Tensor）

### 张量（Tensor）= PyTorch 里专门用来装数字的容器

用最通俗、零基础的话讲：

**张量就是 PyTorch 版的「数字 / 数组」**，

你平时用的整数、小数、列表、表格，在 PyTorch 里统一叫 **张量（Tensor）**。

---

### 按 “维度” 理解，一看就懂

我们从最简单到复杂，一共 4 种：

#### 1）0 维张量（标量）

就是**单个数字**

```python
x = torch.tensor(3.0)
```

就一个数：`3.0`

> 你刚才代码里的 `x` 就是这个。

#### 2）1 维张量（向量）

**一排数字**，像一行列表

```python
x = torch.tensor([1, 2, 3, 4])
```

形状：`(4,)`

#### 3）2 维张量（矩阵）

**行 + 列的表格**，最常用

```python
x = torch.tensor([[1,2],[3,4]])
```

形状：`(2行, 2列)`

#### 4）3 维、4 维张量

比如一张彩色图片 = 长 × 宽 ×3 通道（RGB）

一批图片 = 数量 × 长 × 宽 × 通道

这些在 AI 里非常常见。

---

### 切片（slice）= 从张量里拿出一部分数据

这个和 Python 列表切片很像。

比如：

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
```

这个张量可以理解成：

- 第 0 行是 `[1, 2, 3]`
- 第 1 行是 `[4, 5, 6]`

常见取法：

```python
print(x[0])
print(x[0, 1])
print(x[:, 1])
print(x[0:2, 1:3])
```

分别表示：

- `x[0]` → 取第 0 行，结果是 `tensor([1, 2, 3])`
- `x[0, 1]` → 取第 0 行第 1 列，结果是 `2`
- `x[:, 1]` → 取所有行的第 1 列，结果是 `tensor([2, 5])`
- `x[0:2, 1:3]` → 取第 0 到第 1 行、第 1 到第 2 列

你可以先把它记成：

- 前面的位置表示“行”
- 后面的位置表示“列”
- `:` 表示“这一维我全都要”

---

### `dtype=torch.float32` 是什么意思？

`dtype` 就是：**这个张量里存的数据类型是什么**。

比如：

```python
x = torch.tensor([1, 2, 3], dtype=torch.float32)
```

意思是：

- `x` 是一个张量
- 里面的数据类型是 `float32`
- 也就是 32 位浮点数

为什么经常写 `torch.float32`？

因为深度学习里，大多数输入、权重、计算结果都常用浮点数。

例如：

```python
x = torch.tensor([1, 2, 3], dtype=torch.float32)
print(x)
```

输出会类似：

```python
tensor([1., 2., 3.])
```

你会发现，虽然你写的是 `1, 2, 3`，但输出变成了 `1., 2., 3.`，这就是因为它们已经被当成浮点数了。

常见类型你先知道这几个就够了：

- `torch.float32` → 浮点数，最常用
- `torch.int64` → 整数，常用于标签
- `torch.bool` → 真 / 假

为什么新手阶段要重视这个？

因为有些运算要求数据类型一致。

比如模型参数通常是 `float32`，如果你传进去的是整数张量，有时就会报错，或者结果不符合预期。

所以入门时你可以先养成一个习惯：

**做大多数数值计算时，优先用 `dtype=torch.float32`。**

---

### 关键：为什么不直接用普通数字 / 列表？

普通 Python 数字：

- 不能放 GPU 加速
- 不能自动求梯度（算导数）
- 不能批量做矩阵运算

**张量 = 给 AI 优化过的超级数字**

它能干两件核心事：

1. 可以扔到 **GPU** 上飞快计算
2. 可以开 `requires_grad=True`，让 PyTorch 自动算梯度

---

### 一句话总结

- 普通数字 → PyTorch 里叫：**0 维张量**
- 普通列表 → PyTorch 里叫：**1 维张量**
- 普通表格 → PyTorch 里叫：**2 维张量**

**所有 PyTorch 计算，都必须用张量。**

你之前写的：

```python
x = torch.tensor(3.0, requires_grad=True)
```

就是：

**创建一个张量，并且告诉 PyTorch：我要对它算梯度。**

---

## 第 2 课：自动求导（autograd）

上一课你已经知道：

- `Tensor` 是 PyTorch 的基础对象
- 它像更强大的数组
- 可以指定 `dtype=torch.float32`
- 可以索引和切片

这一课我们学 PyTorch 最核心的能力之一：**自动求导**。

---

### 先理解：为什么要“求导”？

训练模型时，本质上是在做一件事：

- 先让模型算一个结果
- 看结果错了多少
- 再根据“错的方向”调整参数

这个“错的方向”，就靠**导数 / 梯度**来告诉我们。

你先不用把它想得太数学。入门时你可以这样记：

- 梯度 = 告诉参数该往哪个方向改
- 改多少，通常也和梯度有关

所以：

- 没有梯度，就很难训练模型
- PyTorch 的强大之处，就是它能自动帮你算梯度

---

### 最小例子：先看代码

```python
import torch

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x

y.backward()

print(x.grad)
```

输出是：

```python
tensor(8.)
```

---

### 这段代码到底在干嘛？

#### 第一步：创建一个可求导的张量

```python
x = torch.tensor(3.0, requires_grad=True)
```

这里最重要的是：

- `x` 是一个张量
- `requires_grad=True` 表示：我要跟踪它的计算，之后要对它求导

你可以把它理解成：

“PyTorch，请帮我盯住这个变量，后面如果我拿它参与计算，最后我要对它求梯度。”

#### 第二步：定义一个公式

```python
y = x ** 2 + 2 * x
```

这就是数学里的：

```python
y = x² + 2x
```

如果 `x = 3`，那么：

```python
y = 3² + 2*3 = 9 + 6 = 15
```

所以这一步先是在**前向计算**。

#### 第三步：反向传播

```python
y.backward()
```

这一句非常关键。

它的意思是：

- 从 `y` 开始反向求导
- 把对 `x` 的导数算出来
- 结果自动存到 `x.grad` 里

也就是：

```python
dy/dx
```

#### 第四步：查看梯度

```python
print(x.grad)
```

为什么结果是 `8`？

因为：

```python
y = x² + 2x
dy/dx = 2x + 2
```

当 `x = 3` 时：

```python
dy/dx = 2*3 + 2 = 8
```

所以输出：

```python
tensor(8.)
```

---

### `requires_grad=True` 到底有什么用？

如果你不写它：

```python
x = torch.tensor(3.0)
y = x ** 2 + 2 * x
```

那么 PyTorch 会把它当成普通数值计算。

也就是说：

- 能算结果
- 但不会帮你记录求导过程

所以如果后面你调用 `backward()`，通常就不行。

因此你要记住：

- 想求梯度的变量，要加 `requires_grad=True`

---

### `backward()` 是什么？

你可以先这样理解：

- 前向传播：先算出结果
- 反向传播：再从结果往回算梯度

在 PyTorch 里：

```python
y.backward()
```

通常表示：

“请从 `y` 出发，把参与计算且需要梯度的变量的导数都算出来。”

---

### 梯度存在哪里？

答案是：

- 存到 `.grad` 属性里

比如：

```python
print(x.grad)
```

如果 `x` 参与了计算，而且 `requires_grad=True`，在 `backward()` 之后就能看到它的梯度。

---

### `.item()` 是什么？

有时你会看到：

```python
print(x.grad.item())
```

它的作用是：

- 把只有一个值的张量
- 转换成普通 Python 数字

比如：

```python
tensor(8.)
```

变成：

```python
8.0
```

所以：

- `x.grad` 是张量
- `x.grad.item()` 是普通数字

---

### 一个更直观的小例子

```python
import torch

a = torch.tensor(2.0, requires_grad=True)
b = a * 3
c = b + 4

c.backward()

print(a.grad)
```

先手算一下：

```python
c = 3a + 4
dc/da = 3
```

所以输出会是：

```python
tensor(3.)
```

这说明 PyTorch 确实在自动帮你求导。

---

### 这一课你只要记住 4 句话

- `requires_grad=True`：表示这个变量要参与求梯度
- `backward()`：让 PyTorch 开始反向传播
- `.grad`：查看梯度结果
- `.item()`：把单个张量值变成普通 Python 数字







### 重要

```python
import torch
import torch.nn as nn

x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# model 模型本体，负责“根据输入算预测值”
# loss_fn 损失函数，负责“判断模型错了多少”
# optimizer 优化器，负责“根据梯度去更新参数”


for epoch in range(2000):
    optimizer.zero_grad()
    pred = model(x)

    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(epoch, loss.item())
    
print("weight = ", model.weight.item())
print("bias = ", model.bias.item())


```



```python
import torch.nn as nn
```

导入torch.nn模块，并给它起别名 nn， nn是“神经网络工具箱”



```python
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
```

这里有4条样本，并且它的形状是**x.shape = (4, 1)**

**4**表示有4个样本，**1**表示每个样本只有1个特征

**x是输入数据，y是真实答案**



```python
model = nn.Linear(1, 1)
```

这段代码的作用是创建一个线性模型

Linear(1, 1)的含义是

- 输入维度是1
- 输出维度是1

这个模型本质上就是

```python
y = wx + b
```

里面有两个可训练参数：（这两个参数是nn.Linear这个层自动创建的参数）

- `weight`，也就是 `w`
- `bias`，也就是 `b`

**为什么这里写 `1, 1`**

- 因为你的输入每条样本只有 1 个特征
- 输出也只是 1 个数



```python
loss_fn = nn.MSELoss()
```

这段代码的作用是创建损失函数

**`MSELoss` 的意思**

- Mean Squared Error Loss
- 中文叫“均方误差损失”

它负责什么

- 比较模型预测值和真实值之间差多少

直觉理解

- 预测越不准，`loss` 越大
- 预测越准，`loss` 越小

你可以把它理解成

- 给模型当前表现打分的规则。



```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

这段代码是**创建优化器**

SGD是随机梯度下降，最基础、最常见的优化方法之一

model.parameters():把模型里的可训练参数交给优化器管理，也就是weight，bias

lr = 0.01是学习率，表示每次更新参数时迈多大一步



```
for epoch in range(2000):
```

作用：让训练重复进行2000轮



```python
optimizer.zero_gard()
```

作用是：清空上一轮的梯度，因为pytorch默认会把梯度累加，如果不清空，这一轮和上一轮的梯度会混在一起

 



```
pred = model(x)
```

作用：把输入x送进模型，得到预测值pred



也叫

- 前向传播
- forward

本质上模型在做

```python
pred = w*x + b
```

生成出来的pred

```python
pred.shape = (4,1)
```



**只不过**

- 这里的 `w` 和 `b` 是模型当前学到的参数
- 一开始并不准确



```python
loss = loss_fn(pred, y)
```

首先这段代码在算的时候，pred和y是按张量的shape和元素位置一一对应来计算的



### 非常重要

```python
loss.backward()
```

这段代码的作用是：反向传播，自动计算梯度

```python
pred = model(x)
loss = loss_fn(pred, y)
loss.backward()
```

这三步的关系是：

- `model(x)` 先算出预测值 `pred`
- `loss_fn(pred, y)` 再根据预测值和真实值算出误差 `loss`
- `loss.backward()` 就是从这个“误差结果”开始，往回算每个参数的梯度

backward()：沿着计算过程反着走，使用链式法则，把梯度一层一层算回去

计算链

```python
x -> model -> pred -> loss_fn -> loss
```

当调用

```python
loss.backward()
```

```python
loss
  ↓
pred
  ↓
model.weight, model.bias
```

最后把结果存到：

- model.weight
- model.bias

存的结果是

```
Loss对model.weight的梯度和Loss对model.bias的梯度
```



```python
optimizer.step()
```

这段代码的作用是：让优化器根据刚刚算出的梯度更新参数

参数是：

- weight
- bias

这两个参数是根据

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
model.parameters():把模型里的可训练参数交给优化器管理，也就是weight，bias
```

