
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import csv
import pandas as pd
from tensorflow.python.platform import gfile


# In[2]:


#Load data
with gfile.Open('../test.csv') as csv_file:
    data_file = csv.reader(csv_file)
    data = []
    for row in data_file:
        data.append(np.asarray(row, dtype=np.float32))
x_test = np.array(data)
x=tf.reshape(x_test/255,[-1,28,28,1])


# In[4]:


#Load LeNet-5 Model
saver=tf.train.import_meta_graph('LeNet-5-150.meta')
graph = tf.get_default_graph()
sess=tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess,'LeNet-5-150')
C1_filter=graph.get_tensor_by_name('C1_filter:0')
C1_bias=graph.get_tensor_by_name('C1_bias:0')
C3_filter=graph.get_tensor_by_name('C3_filter:0')
C3_bias=graph.get_tensor_by_name('C3_bias:0')
C5_filter=graph.get_tensor_by_name('C5_filter:0')
C5_bias=graph.get_tensor_by_name('C5_bias:0')
F6_weights=graph.get_tensor_by_name('F6_weights:0')
F6_bias=graph.get_tensor_by_name('F6_bias:0')
y_weights=graph.get_tensor_by_name('y_weights:0')
y_bias=graph.get_tensor_by_name('y_bias:0')


# In[5]:


#LeNet-5 structure
C1 = tf.nn.conv2d(x, C1_filter,strides=[1, 1, 1, 1], padding='SAME')
D1 = tf.nn.relu(C1+C1_bias)
S2 = tf.nn.max_pool(D1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
C3 = tf.nn.conv2d(S2, C3_filter,strides=[1, 1, 1, 1], padding='VALID')
D3 = tf.nn.relu(C3+C3_bias)
S4 = tf.nn.max_pool(D3, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
C5 = tf.nn.conv2d(S4, C5_filter,strides=[1, 1, 1, 1], padding='VALID')
D5_pre = tf.nn.relu(C5+C5_bias)
D5=tf.reshape(D5_pre,[-1,120])
F6=tf.nn.relu(tf.matmul(D5, F6_weights)+F6_bias)
y=tf.nn.softmax(tf.matmul(F6, y_weights)+y_bias)


# In[6]:


y_out=tf.argmax(y,1)


# In[7]:


out=sess.run(y_out)


# In[8]:


print(type(out))


# In[9]:


pd.DataFrame({"ImageId": range(1, len(out)+1), "Label": out}).to_csv('out.csv', index=False, header=True)

