
# coding: utf-8

# In[1]:


import numpy as np
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# In[2]:


class VFLearn:
    def __init__(self, featureLength, input_var, train_mode, config, agent):
        
        # Configuration details object
        self.config = config
        
        # Length of state input
        self.feats = featureLength
        
        # Placeholder for input
        self.input_var = input_var 

        # Placeholder for training or eval mode.
        self.train_mode = train_mode 
        
        # which agent?
        self.agent = agent

        # Output tensor
        self.l_vf = self.initNNVF(drate=self.config.drate)     

#         self.obs_var = TT.dmatrix(name='states')
#         self.returns_var = TT.vector('returns')
#         self.learningrate = self.config.learningRate
#         self.updates = None
        
    def initNNVF(self, drate):
        
        l_in = tf.layers.dense(inputs=self.input_var, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLu_1_Baseline_Agent_"+str(self.agent))
        dropout1 = tf.layers.dropout(inputs=l_in, rate=drate, training=self.train_mode, name="Dropout_1_Baseline_Agent_"+str(self.agent))

        l_hid2 = tf.layers.dense(inputs=dropout1, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLu_2_Baseline_Agent_"+str(self.agent))
        dropout2 = tf.layers.dropout(inputs=l_hid2, rate=drate, training=self.train_mode, name="Dropout_2_Baseline_Agent_"+str(self.agent))

        l_hid3 = tf.layers.dense(inputs=dropout2, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLu_3_Baseline_Agent_"+str(self.agent))
        dropout3 = tf.layers.dropout(inputs=l_hid3, rate=drate, training=self.train_mode, name="Dropout_3_Baseline_Agent_"+str(self.agent))

        l_hid4 = tf.layers.dense(inputs=dropout3, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLu_4_Baseline_Agent_"+str(self.agent))
        dropout4 = tf.layers.dropout(inputs=l_hid4, rate=drate, training=self.train_mode, name="Dropout_4_Baseline_Agent_"+str(self.agent))
        
        l_out = tf.layers.dense(inputs=dropout4, units=1, activation=None, name="Dense_5_Baseline_Agent_"+str(self.agent))

        print "===Successfully initiated a neural network===\n"
        print('input_Shape', self.input_var.get_shape())
        print('out_Shape', l_out.get_shape())

        return l_out
    
    # Gives the value function of input states.
    # Expected Shape = N*S
    def get_value(self, flat_obs, sess):
        val = self.l_vf.eval({self.input_var: flat_obs, self.train_mode: False}, session=sess)
        return val


# In[6]:


# Testing for the network computational graph.
# input_var = tf.placeholder(shape=[None, 5], dtype=tf.float32)
# train_mode = tf.placeholder(tf.bool)
# test = VFLearn(featureLength=5, input_var=input_var, train_mode=train_mode, config=None)

# x = [[1,2,3,4,5]]
# x.append([6,7,8,9,10])
# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     training_mode = False
#     out = sess.run(test.l_vf, {test.input_var: x, test.train_mode: training_mode})
#     print out, out.shape
#     print test.get_value(flat_obs=x)
    
#     # Testing the train_TD method
    
#     obs = [[1,2,3,4,5]]
#     obs.append([6,7,8,9,10])
#     obs = np.array(obs)
    
#     rew = [4,5]
#     rew = np.array(rew)
    
#     new_obs = [[7,8,9,10,11]]
#     new_obs.append([7,3,4,2,1])
#     new_obs = np.array(new_obs)
    
    #test.train_TD(obs, rew, new_obs, sess)

