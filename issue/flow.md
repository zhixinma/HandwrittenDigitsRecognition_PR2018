# Tensorflow flow

----
## 1. Build the Graph (statistic)
#### *1). Interface*
尽可能地构建图表，做到返回包含了预测结果（output prediction）的Tensor


#### *2). Loss*
通过添加所需的损失操作，进一步构建图表

#### *3). Train*
添加了通过梯度下降（gradient descent）将损失最小化所需的操作

## 2. Model Training (dynamic)
一旦图表构建完毕，就通过 `fully_connected_feed.py` 文件中的用户代码进行循环地迭代式训练和评估。

#### *1).Graph*
在run_training()这个函数的一开始，是一个Python语言中的with命令，这个命令表明所有已经构建的操作都要与默认的tf.Graph全局实例关联起来。
```python
with tf.Graph().as_default():
```
tf.Graph实例是一系列可以作为整体执行的操作。TensorFlow的大部分场景只需要依赖默认图表一个实例即可。

#### *2).Session*
完成全部的构建准备、生成全部所需的操作之后，我们就可以创建一个 `tf.Session` ，用于运行图表。
```python
sess = tf.Session()
# 另外，也可以利用with代码块生成Session，限制作用域：

with tf.Session() as sess:
# Session函数中没有传入参数，表明该代码将会依附于（如果还没有创建会话，则会创建新的会话）默认的本地会话。
```

#### *3).Training Loop*
完成会话中变量的初始化之后，就可以开始训练了。
```python
# 训练的每一步都是通过用户代码控制，而能实现有效训练的最简单循环就是：
for step in xrange(max_steps):
    sess.run(train_op)
# 但是，本教程中的例子要更为复杂一点，原因是我们必须把输入的数据根据每一步的情况进行切分，以匹配之前生成的占位符。
```
#### *4).Feedback*
```python
# 执行每一步时，我们的代码会生成一个反馈字典（feed dictionary）
# 其中包含对应步骤中训练所要使用的例子，这些例子的哈希键就是其所代表的占位符操作。
```
#### *5).Check the states*
在运行 `sess.run` 函数时，要在代码中明确其需要获取的两个值：`[train_op, loss]`
```python
for step in xrange(FLAGS.max_steps):
    feed_dict = fill_feed_dict(data_sets.train,
                               images_placeholder,
                               labels_placeholder)
    _, loss_value = sess.run([train_op, loss],
                             feed_dict=feed_dict)
```
#### *6).States Visualization*
为了释放TensorBoard所使用的事件文件（events file），所有的即时数据都要在**图表构建阶段**合并至一个op中。
```python
summary_op = tf.merge_all_summaries()
```
在创建好Session之后，可以实例化一个 `tf.train.SummaryWriter` ，用于写入包含了图表本身和即时数据具体值的事件文件。
```python
summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                        graph_def=sess.graph_def)
```
最后，**每次运行summary_op**时，都会往事件文件中写入最新的即时数据，***函数(summary_op)的输出会传入事件文件读写器***（writer）的add_summary()函数。
```python
# cary out summary_op() and return the latest data
summary_str = sess.run(summary_op, feed_dict=feed_dict)
# merge data, but what's the contribution of step
summary_writer.add_summary(summary_str, step)
```
事件文件写入完毕之后，可以就训练文件夹打开一个TensorBoard，查看即时数据的情况。

#### *7).Checkpoint*.
为了得到可以用来后续恢复模型以进一步训练或评估的检查点文件（checkpoint file），我们实例化一个tf.train.Saver。

saver = tf.train.Saver()
在训练循环中，将定期调用saver.save()方法，向训练文件夹中写入包含了当前所有可训练变量值得检查点文件。
```python
saver.save(sess, FLAGS.train_dir, global_step=step)
```
这样，我们以后就可以使用saver.restore()方法，重载模型的参数，继续训练。
```python
saver.restore(sess, FLAGS.train_dir)
```
#### *8).Evaluate our Model*
`do_eval()`

#### *9).Eval Graph*
#### *10).Eval Output*
在打开默认图表（Graph）之前，我们应该先调用get_data(train=False)函数，抓取测试数据集。
```python
test_all_images, test_all_labels = get_data(train=False)
```
