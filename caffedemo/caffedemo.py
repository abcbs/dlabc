#encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

#设置默认显示参数
plt.rcParams['figure.figsize'] = (10, 10)        # 图像显示大小
plt.rcParams['image.interpolation'] = 'nearest'  # 最近邻差值: 像素为正方形
plt.rcParams['image.cmap'] = 'gray'  # 使用灰度输出而不是彩色输出

# caffe模块要在Python的路径下;
# 这里我们将把caffe 模块添加到Python路径下.
import sys
caffe_root = 'G:\\win_tool_prog\\caffe\\'  #该文件要从路径{caffe_root}/examples下运行，否则要调整这一行。
sys.path.insert(0, caffe_root + 'python')

import caffe
# 如果你看到"No module named _caffe",那么要么就是你没有正确编译pycaffe；要么就是你的路径有错误。
import os
if os.path.isfile('D:/DevN/abc-dl-dn/' + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print( 'CaffeNet found.')
else:
    print ('Downloading pre-trained CaffeNet model...')
    os.system('D:/DevN/abc-dl-dn/scripts/download_model_binary.py D:/DevN/abc-dl-dn//models/bvlc_reference_caffenet ')

caffe.set_mode_cpu()

model_def =  'D:/DevN/abc-dl-dn/models/bvlc_reference_caffenet/deploy.prototxt'
model_weights =  'D:/DevN/abc-dl-dn/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(model_def,      # 定义模型结构
                model_weights,  # 包含了模型的训练权值
                caffe.TEST)     # 使用测试模式(不执行dropout)

# 加载ImageNet图像均值 (随着Caffe一起发布的)
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  #对所有像素值取平均以此获取BGR的均值像素值
print ('mean-subtracted values:', zip('BGR', mu))

# 对输入数据进行变换
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  #将图像的通道数设置为outermost的维数
transformer.set_mean('data', mu)            #对于每个通道，都减去BGR的均值像素值
transformer.set_raw_scale('data', 255)      #将像素值从[0,255]变换到[0,1]之间
transformer.set_channel_swap('data', (2,1,0))  #交换通道，从RGB变换到BGR

# 设置输入图像大小
net.blobs['data'].reshape(50,        # batch 大小
                          3,         # 3-channel (BGR) images
                          227, 227)  # 图像大小为:227x227
image = caffe.io.load_image( 'G:/win_tool_prog/caffe-1.0/examples/examples/images/cat.jpg')
transformed_image = transformer.preprocess('data', image)
plt.imshow(image)
plt.show()