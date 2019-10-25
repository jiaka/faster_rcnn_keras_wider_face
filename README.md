# faster_rcnn_keras_wider_face
用Faster RCNN实现人脸检测（基于wider_face数据集）
### 一，文件介绍
1. [train__test.ipynb](https://github.com/jiaka/faster_rcnn-keras-_VOC2012/blob/master/faster_rcnn_train_and_test.ipynb)是主要的入口文件，进行参数初始化，训练数据处理，网络搭建与训练，网络预测。
2. [resnet50.py](https://github.com/jiaka/faster_rcnn-keras-_VOC2012/blob/master/resnet50.py)定义了网络结构的函数，分类网络和回归网络。
3. [roi_helpers.py](https://github.com/jiaka/faster_rcnn-keras-_VOC2012/blob/master/roi_helpers.py)可以获得筛选后的预选框，交并比，将rpn预测结果转化为预选框等。
4. [losses.py](https://github.com/jiaka/faster_rcnn-keras-_VOC2012/blob/master/losses.py)定义了各个损失函数。
5. [data_generator.py](https://github.com/jiaka/faster_rcnn-keras-_VOC2012/blob/master/data_generator.py)包括了图片增强，得到rpn网络的训练数据，anchor属性计算等。
6. [data_augment.py](https://github.com/jiaka/faster_rcnn-keras-_VOC2012/blob/master/data_augment.py)为图片增强处理函数，包括对图片的翻转或旋转。
7. [RoiPoolingConv.py](https://github.com/jiaka/faster_rcnn-keras-_VOC2012/blob/master/RoiPoolingConv.py)定义roipooling层，将每个图片整合到相同大小。
8. [FixedBatchNormalizations.py](https://github.com/jiaka/faster_rcnn-keras-_VOC2012/blob/master/FixedBatchNormalizations.py)相当于重新定义了Batch Normalization。

### 二，资料下载
  本项目采用的数据集是wider face数据集，官方下载地址：[http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)

  该网络采用resnet50网络，该网络的权重文件下载地址是：[https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5](https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5)
  
  百度网盘下载地址：链接: [https://pan.baidu.com/s/18BL8z3jH-dUnLaKe6sa4sw&shfl=shareset](https://pan.baidu.com/s/18BL8z3jH-dUnLaKe6sa4sw&shfl=shareset) 提取码: mexy 

  
### 三，使用说明
  1. 锚框的大小为[128、256、512]，比率为[1：1、1：2、2：1]。
  2. tensorflow的版本是'1.9.0',keras的版本是'2.1.5',除了使用tensorflow2.0之后版本，其他版本都可以尝试。不支持python2.x。
  3. 使用的是tensorflow backend，theano可以自行修改。
  4. wider face的Label文件格式与VOC2012的label不同，而我使用的Faster RCNN需要VOC2012的格式，所以需要将label文件转换一下格式。具体可以查看
  [https://blog.csdn.net/qq_37431083/article/details/102742322](https://blog.csdn.net/qq_37431083/article/details/102742322)
  5. 在训练过程中可能会出现`"ValueError: 'a' cannot be empty unless no samples are taken"`这个错误，原因是neg_samples（负样本）的值是一个空值，一般在一轮训练中可能会有负样本为空的情况，这里直接用异常处理跳过了，如果有更好的办法欢迎指正。
  [https://blog.csdn.net/qq_37431083/article/details/102628580](https://blog.csdn.net/qq_37431083/article/details/102628580)
  6. 这里只使用了wider face数据集中的训练集部分，将训练集分割成训练集，验证集，测试集，但主要使用的是训练集。
  
### 四，预测结果
  由于显卡性能有限，这里只训练了30轮左右的预测结果，挑出了比较好的一张。
  [https://github.com/jiaka/faster_rcnn_keras_wider_face/blob/master/results_imgs/10.png](https://github.com/jiaka/faster_rcnn_keras_wider_face/blob/master/results_imgs/10.png)
  
### 五，参考资料
1. [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf)
2. [https://github.com/yhenon/keras-frcnn/](https://github.com/yhenon/keras-frcnn/)
