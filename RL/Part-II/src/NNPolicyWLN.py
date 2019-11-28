
# coding: utf-8

# In[1]:


import numpy as np
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# In[2]:


TINY = 1e-8


# In[4]:


class ANNPolWLN:
    def __init__(self, featureLength, actionCount, input_var, train_mode, agent, config):
        # Length of input state
        self.feats = featureLength
        
        # Number of actions, i.e. output units
        self.act = actionCount
        
        # Placeholder for input
        self.input_var = input_var 

        # Placeholder for training or eval mode.
        self.train_mode = train_mode
        
        # Which agent?
        self.agent = agent
        
        # Config Instance
        self.config = config

        # Output tensor
        self.l_prob = self.initNNPolicy()

    # Remember to change the mode while evaluating!
    def initNNPolicy(self, drate=0.0):
        
        norm1 = tf.contrib.layers.layer_norm(inputs=self.input_var, trainable=True, scope="NormLayer_1_Policy_Agent_"+str(self.agent))
        l_in = tf.layers.dense(inputs=norm1, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLu_1_Policy_Agent_"+str(self.agent))
        dropout1 = tf.layers.dropout(inputs=l_in, rate=drate, training=self.train_mode, name="Dropout_1_Policy_Agent_"+str(self.agent))

        norm2 = tf.contrib.layers.layer_norm(inputs=dropout1, trainable=True, scope="NormLayer_2_Policy_Agent_"+str(self.agent))
        l_hid2 = tf.layers.dense(inputs=norm2, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLu_2_Policy_Agent_"+str(self.agent))
        dropout2 = tf.layers.dropout(inputs=l_hid2, rate=drate, training=self.train_mode, name="Dropout_2_Policy_Agent_"+str(self.agent))

        norm3 = tf.contrib.layers.layer_norm(inputs=dropout2, trainable=True, scope="NormLayer_3_Policy_Agent_"+str(self.agent))
        l_hid3 = tf.layers.dense(inputs=norm3, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLu_3_Policy_Agent_"+str(self.agent))
        dropout3 = tf.layers.dropout(inputs=l_hid3, rate=drate, training=self.train_mode, name="Dropout_3_Policy_Agent_"+str(self.agent))

        norm4 = tf.contrib.layers.layer_norm(inputs=dropout3, trainable=True, scope="NormLayer_4_Policy_Agent_"+str(self.agent))
        l_hid4 = tf.layers.dense(inputs=norm4, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLu_4_Policy_Agent_"+str(self.agent))
        dropout4 = tf.layers.dropout(inputs=l_hid4, rate=drate, training=self.train_mode, name="Dropout_4_Policy_Agent_"+str(self.agent))
        
        norm5 = tf.contrib.layers.layer_norm(inputs=dropout4, trainable=True, scope="NormLayer_5_Policy_Agent_"+str(self.agent))
        logits = tf.layers.dense(inputs=norm5, units=self.act, name="Dense_5_Policy_Agent_"+str(self.agent))

        l_out = tf.nn.softmax(logits, name="Softmax_Tensor_Policy_Agent_"+str(self.agent))

        print "\n===Successfully initiated a neural network==="
        print('input_Shape', self.input_var.get_shape())
        print('out_Shape', l_out.get_shape())

        return l_out

    # Take as input flat observation or state
    # Determines what action to take in current state depending on current parameter values.
    # Return (sampeldAction, actionProb)
    def get_action(self, flat_obs, sess):
        prob = self.l_prob.eval({self.input_var: flat_obs, self.train_mode: False}, session=sess)
        #sample = np.random.multinomial(1, prob[0])
        #act = np.argmax(sample)
        act = np.random.choice(self.act, 1, p=prob[0])
        return (act, prob)
    
    # Get the symbolic variables for <pi(a|s)>
    # Dimension N*A
    def action_info_sym(self, obs_var):
        out = self.l_prob.eval({self.input_var: obs_var, self.train_mode: True})
        return dict(prob=out)
    
    # Input symbolic variables: <actions, obs>
    # "actions" is dimension N*A, obs is N*featureLength
    # Returns N*1: log pi(a|s)
    def log_likeli_sym(self, actions):
        return tf.reduce_sum(tf.multiply(actions, tf.log(self.l_prob)), 1) + TINY
        # probs = self.action_info_sym(obs)["prob"]


# In[10]:


# # Testing for the network computational graph.

# arrays = [3,2,3]
# arrayfd = [3,5,8]

# x = [[1,2,3,4,5]]
# x.append([6,7,8,9,10])
# x.append([7,4,3,2,1])

# y = [[6,5,3,2,1]]
# y.append([9,8,7,5,4])

# z = [[6,5,8,9,2]]
# z.append([7,8,9,4,3])
# z.append([5,4,2,2,2])

# final = []
# final.extend(x)
# final.extend(y)
# final.extend(z)
# final = np.array(final).reshape((arrayfd[-1], 5))

# a = [[0.,0.,1.]]
# a.extend([[1.,0.,0.]]*7)
# print a
# a = np.array(a, dtype=np.float32).reshape((arrayfd[-1], 3))

# g = tf.Graph()
# with g.as_default():
    
#     arrays = tf.placeholder(shape=[3,1], dtype=tf.int32, name="Hello")
#     input_var = tf.placeholder(shape=[None, 5], dtype=tf.float32)
#     train_mode = tf.placeholder(tf.bool)
#     test = ANNPol(featureLength=5, actionCount=3, input_var=input_var, train_mode=train_mode)
    
#     act_var = tf.placeholder(shape=[None, 3], dtype=tf.float32)
#     all_outputs = test.log_likeli_sym(act_var)
#     all_outputs = tf.reshape(all_outputs, shape=[8,1])
    
#     loss1 = all_outputs[0:arrays[0,0]]
#     loss2 = all_outputs[arrays[1,0]:arrays[1,0]]
#     loss3 = all_outputs[arrays[1,0]:arrays[2,0]]
# #     loss2 = all_outputs[arrays[0]:arrays[1]]
# #     loss3 = all_outputs[arrays[1]:arrays[2]]
    

    
# with tf.Session(graph=g) as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
    
#     dictd = dict()
#     dictd[arrays] = np.array(arrayfd).reshape((3,1))
#     print dictd[arrays]
#     dictd[input_var] = final
#     dictd[act_var] = a
#     dictd[test.train_mode] = False
    
#     print sess.run(all_outputs, feed_dict=dictd)
#     print sess.run(loss1, feed_dict=dictd)
#     print sess.run(tf.reduce_mean(loss2)*0, feed_dict=dictd)
#     print sess.run(loss3, feed_dict=dictd)


