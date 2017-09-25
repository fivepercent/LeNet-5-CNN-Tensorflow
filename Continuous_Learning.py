
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np


# In[ ]:


saver=tf.train.import_meta_graph('LeNet-5-150.meta')
graph = tf.get_default_graph()
sess=tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess,'LeNet-5-150')
cross_entropy=graph.get_tensor_by_name(sess, 'Neg:0')


# In[ ]:


train=tf.train.AdamOptimizer(0.01).minimize(cross_entropy)


# In[ ]:


for step in range(111, 151):
    sess.run(train)
    if(step%10==0):
        print (step, sess.run(cross_entropy), sess.run(accuracy))


# In[ ]:


saver=tf.train.Saver()
saver.save(sess, 'LeNet-5', global_step=150)

