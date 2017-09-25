
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np


# In[2]:


#Load data
MNIST_TRAIN = "../train.csv"

training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=MNIST_TRAIN,
    target_column=0,
    target_dtype=np.int,
    features_dtype=np.float32)


# In[3]:


#data pre-process
x_train=training_set.data/255
train_target=training_set.target
m=train_target.shape[0]
y_train=tf.one_hot(train_target, 10)
x = tf.reshape(x_train,[-1,28,28,1])


# In[5]:


def weights(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1),name=name)


# In[6]:


#LeNet-5 structure
C1_filter=weights([5,5,1,6],'C1_filter')#filter_height, filter_width, in_channels, out_channels
C1_bias=tf.Variable(tf.constant(0.1, shape=[6]),name='C1_bias')
C1 = tf.nn.conv2d(x, C1_filter,strides=[1, 1, 1, 1], padding='SAME')
D1 = tf.nn.relu(C1+C1_bias)

S2 = tf.nn.max_pool(D1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

C3_filter=weights([5,5,6,16], 'C3_filter')#filter_height, filter_width, in_channels, out_channels
C3_bias=tf.Variable(tf.constant(0.1, shape=[16]), name='C3_bias')
C3 = tf.nn.conv2d(S2, C3_filter,strides=[1, 1, 1, 1], padding='VALID')
D3 = tf.nn.relu(C3+C3_bias)

S4 = tf.nn.max_pool(D3, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

C5_filter=weights([5,5,16,120], 'C5_filter')#filter_height, filter_width, in_channels, out_channels
C5_bias=tf.Variable(tf.constant(0.1, shape=[120]), name='C5_bias')
C5 = tf.nn.conv2d(S4, C5_filter,strides=[1, 1, 1, 1], padding='VALID')
D5_pre = tf.nn.relu(C5+C5_bias)
D5=tf.reshape(D5_pre,[-1,120])

F6_weights=weights([120,84], 'F6_weights')
F6_bias=tf.Variable(tf.constant(0.1, shape=[84]), name='F6_bias')
F6=tf.nn.relu(tf.matmul(D5, F6_weights)+F6_bias)

y_weights=weights([84,10],'y_weights')
y_bias=tf.Variable(tf.constant(0.1, shape=[10]), name='y_bias')
y=tf.nn.softmax(tf.matmul(F6, y_weights)+y_bias)


# In[7]:


#cost function and accuracy
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y_train*tf.log(y),1))
train=tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_train, 1))
accuracy= tf.reduce_mean(tf.cast(correct, tf.float32))


# In[8]:


sess=tf.Session()
sess.run(tf.global_variables_initializer())


# In[16]:


#training
for step in range(1, 151):
    sess.run(train)
    if(step%10==0):
        print (step, sess.run(cross_entropy), sess.run(accuracy))


# In[17]:


#Save model for continuous learning
saver=tf.train.Saver()
saver.save(sess, 'LeNet-5', global_step=150)


# In[18]:


sess.close()


# In[ ]:




