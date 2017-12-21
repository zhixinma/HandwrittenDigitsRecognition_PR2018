## Some problems when install TensorFlow

---

***1.***
>Missing dependencies for SOCKS support when pip install

we can print our dependencies by following command :
`pip list`

Unset socks proxy, in your case:
`unset all_proxy`
`unset ALL_PROXY`

if there are not 'pysocks', you need to install the missing dependency
`pip install pysocks`


***2.***
>No module named contrib.learn.python.learn.datasets.mnist

It's because that your tensorflow's version is too old,
you can uninstall the old version
`sudo pip uninstall tensorflow`

then download a latest version
`sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0-cp27-none-linux_x86_64.whl`

***3.***
If it's too slow to install tensorflow with pip, or even time out
you can download `tensorflow*.whl` and install it locally
`wheel install tensorflow*.whl`
`pip install tensorflow*.whl`

***4.***
And then, a new problem has arisen...

error log
>Traceback (most recent call last):
  File "cnn.py", line 3, in <module>
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py", line 211, in read_data_sets
    SOURCE_URL + TRAIN_IMAGES)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/contrib/learn/python/learn/datasets/base.py", line 141, in maybe_download
    urllib.request.urlretrieve(source_url, temp_file_name)
  File "/usr/lib/python2.7/urllib.py", line 98, in urlretrieve
    return opener.retrieve(url, filename, reporthook, data)
  File "/usr/lib/python2.7/urllib.py", line 273, in retrieve
    block = fp.read(bs)
  File "/usr/lib/python2.7/socket.py", line 384, in read
    data = self._sock.recv(left)
KeyboardInterrupt



```python
# /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
local_file = base.maybe_download(TRAIN_IMAGES, train_dir,
                                   SOURCE_URL + TRAIN_IMAGES)
```
```python
# /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/learn/python/learn/datasets/base.py
def maybe_download(filename, work_directory, source_url):
  """Download the data from source url, unless it's already here.

  Args:
      filename: string, name of the file in the directory.
      work_directory: string, path to working directory.
      source_url: url to download from if file doesn't exist.

  Returns:
      Path to resulting file.
  """
  if not gfile.Exists(work_directory):
    gfile.MakeDirs(work_directory)
  filepath = os.path.join(work_directory, filename)
  if not gfile.Exists(filepath):
    with tempfile.NamedTemporaryFile() as tmpfile:
      temp_file_name = tmpfile.name
      urllib.request.urlretrieve(source_url, temp_file_name)
      gfile.Copy(temp_file_name, filepath)
      with gfile.GFile(filepath) as f:
        size = f.size()
      print('Successfully downloaded', filename, size, 'bytes.')
 return filepath

```

so filepath should be `MNIST_data/*.gz`
Although the problem has been solved, but why maybe_download() doesn't work?

How cand we save parameters of model:
tf.Saver(sess, file_path)

Why the `\tmp\model.ckpt` changed into three file ?
tf.saver

***8.***
ValueError: Unable to determine SOCKS version from socks://127.0.0.1:1080/
No matching distribution found for tensorflow-tensorboard

***9.***
saver = import_meta_graph()
saver.restore(sess, file_path)
