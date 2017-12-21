## Tensorflow Function

---

***tf.size()***

***tf.expand_dims()***

***tf.shape()***

***tf.range(start, limit, delta) / tf.range(limit)***
```python
start = 3
limit = 18
delta = 3
# [3, 6, 9, 12, 15]
```

```python
start = 3
limit = 1
delta = -0.5
# [3, 2.5, 2, 1.5]
```

```python
limit = 5
# [0, 1, 2, 3, 4]
```
***sparse_to_dense()***

***tf.pack() -> tf.stack()***
```python
x = tf.constant([1, 4])
y = tf.constant([2, 5])
z = tf.constant([3, 6])
tf.stack([x, y, z])  # [[1, 4], [2, 5], [3, 6]] (Pack along first dim.)
tf.stack([x, y, z], axis=1)  # [[1, 2, 3], [4, 5, 6]]
```
***concat(
    values,
    axis,
    name='concat'
)***
```python
t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

# tensor t3 with shape [2, 3]
# tensor t4 with shape [2, 3]
tf.shape(tf.concat([t3, t4], 0))  # [4, 3]
tf.shape(tf.concat([t3, t4], 1))  # [2, 6]
```
***tf.scalar_summary(loss.op.name, loss)***

首先，该函数从loss()函数中获取损失Tensor，将其交给tf.scalar_summary，后者在与SummaryWriter配合使用时，可以向事件文件（events file）中生成汇总值（summary values）。
在本篇教程中，每次写入汇总值时，它都会释放损失Tensor的当前值（snapshot value）。

***tf.train.GradientDescentOptimizer***
```python
# 之后，我们生成一个变量用于保存全局训练步骤（global training step）的数值
# 并使用minimize()函数更新系统中的三角权重（triangle weights）、增加全局步骤的操作
# 根据惯例，这个操作被称为 train_op，是TensorFlow会话（session）诱发一个完整训练步骤所必须运行的操作
optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = optimizer.minimize(loss, global_step=global_step)
# 最后，程序返回包含了训练操作（training op）输出结果的Tensor
```
