#encoding=utf-8
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as  mpatches
import caffe
# %matplotlib inline

# 设置默认的属性：用于在ipython中显示图片
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
from math import pow
from skimage import transform as tf

caffe_root='G:\\win_tool_prog\\caffe\\'
sys.path.insert(0, caffe_root + 'python')
# sys.insert.path(0,caffe_root+'python')

caffe_modelcaffe=caffe_root+''
caffe_deploy=caffe_root+''

caffe.set_mode_cpu()
net=caffe.Net(caffe_deploy,caffe_modelcaffe,caffe.TEST)


transform=caffe.io.Transformer({'data':net.blobs['data'].data.shape})
transform.set_transpose('data',(2,0,1))
transform.set_raw_scale('data',255)
transform.set_channel_swap('data',(2,1,0))

#把加载到的图片缩放到固定的大小
net.blobs['data'].reshape(1,2,227,227)

image=caffe.io.load_image('G:\\迅雷下载\\caffe-1.0\\examples\\images\\cat gray.jpg')
transformed_image=transform.preprocess('data',image)
plt.inshow(image)

# 把警告过transform.preprocess处理过的图片加载到内存
net.blobs['data'].data[...]=transformed_image

output=net.forward()

#因为这里仅仅测试了一张图片
#output_pro的shape中有对于1000个object相似的概率
output_pro=output['prob'][0]

#从候选的区域中找出最有可能的那个object的索引
output_pro_max_index=output_pro.argmax()

labels_file = caffe_root + '.../synset_words.txt'
if not os.path.exists(labels_file):
    print ("in the direct without this synset_words.txt ")

labels=np.loadtxt(labels_file,str,delimiter='\t')

# 从对应的索引文件中找到最终的预测结果
outpur_label=labels[output_pro_max_index]
# 也可以找到排名前五的预测结果
top_five_index=output_pro.argsort()[::-1][:5]
print ('probabilities and labels:')
zip(output_pro[top_five_index],labels[top_five_index])