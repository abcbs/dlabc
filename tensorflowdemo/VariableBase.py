# -*- coding: UTF-8 -*-
__author__ = 'liujianqiang'
import tensorflow as tf

sess = tf.InteractiveSession()
x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])
# 使用初始化器 initializer op 的 run() 方法初始化 'x'
x.initializer.run()
# 增加一个减法 sub op, 从 'x' 减去 'a'. 运行减法 op, 输出结果
sub = tf.subtract(x, a)
print (sub.eval())

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul4 = tf.multiply(input1, intermed)

with tf.Session():
    result = sess.run([mul4, intermed])
    print (result)
