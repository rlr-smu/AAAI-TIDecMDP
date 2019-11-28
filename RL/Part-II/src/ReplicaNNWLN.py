
# coding: utf-8

# In[ ]:


import numpy as np
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# In[ ]:


class ReplicaNNWLN:

    def __init__(self, featureSAS, featsSAS_Events, actionCount, input_var, agent, config):
        
        # Config
        self.config = config
                
        # Number of features for encoder
        self.feats_SAS = featureSAS
        
        self.featsSAS_Events = featsSAS_Events
        
        # Number of actions, i.e. output units for m_sa network
        self.act = actionCount
        
        # Placeholder for input, should be (s) from actual NN
        self.input_var = input_var 
        
        # Which agents' relpica?
        self.agent = agent

        # Output tensor
        self.replica_msa_values, self.replica_msa_events_values = self.initReplicaNN()
    
    def initReplicaNN(self, drate=0.0):
        replica_msa_values = []
        replica_msa_events_values = []
        
        MSAR_norm1 = tf.contrib.layers.layer_norm(inputs=self.input_var, trainable=True, scope="NormLayer_1_MSA_Agent_"+str(self.agent))
        MSAR_l_hid_1 = tf.layers.dense(inputs=MSAR_norm1, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_1_MSA_Agent_"+str(self.agent))

        MSAR_norm2 = tf.contrib.layers.layer_norm(inputs=MSAR_l_hid_1, trainable=True, scope="NormLayer_2_MSA_Agent_"+str(self.agent))
        MSAR_l_hid_2 = tf.layers.dense(inputs=MSAR_norm2, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_2_MSA_Agent_"+str(self.agent))

        MSAR_norm3 = tf.contrib.layers.layer_norm(inputs=MSAR_l_hid_2, trainable=True, scope="NormLayer_3_MSA_Agent_"+str(self.agent))
        MSAR_l_hid_3 = tf.layers.dense(inputs=MSAR_norm3, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_3_MSA_Agent_"+str(self.agent))
        
        print "\n===Successfully initiated MSA Replica neural network for agent "+str(self.agent)+"==="
        print('MSAR_Input_Shape', self.input_var.get_shape())

        for i in xrange(0, self.act):
        
            ################################
            # Action i
            ################################ 
            replica_msa_valuesM, replica_msa_valuesE = self.ReplicaM_sa_Network(MSAR_l_hid_3, i)
            replica_msa_values.append(replica_msa_valuesM)
            replica_msa_events_values.append(replica_msa_valuesE)
        
        return replica_msa_values, replica_msa_events_values
    
    def ReplicaM_sa_Network(self, input_v, action):

        MSAR_norm4 = tf.contrib.layers.layer_norm(inputs=input_v, trainable=True, scope="NormLayer_4_Action_"+str(action)+"_MSA_Agent_"+str(self.agent))
        MSAR_l_hid_4 = tf.layers.dense(inputs=MSAR_norm4, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_4_Action_"+str(action)+"_MSA_Agent_"+str(self.agent))

        MSAR_norm5 = tf.contrib.layers.layer_norm(inputs=MSAR_l_hid_4, trainable=True, scope="NormLayer_5_Action_"+str(action)+"_MSA_Agent_"+str(self.agent))
        MSAR_out = tf.layers.dense(inputs=MSAR_norm5, units=self.feats_SAS, activation=None, trainable=True, name="DenseLayer_5_Action_"+str(action)+"_MSA_Agent_"+str(self.agent))

        MSAREv_norm4 = tf.contrib.layers.layer_norm(inputs=input_v, trainable=True, scope="NormLayer_4_Action_"+str(action)+"_Events_MSA_Agent_"+str(self.agent))
        MSAREv_l_hid_4 = tf.layers.dense(inputs=MSAREv_norm4, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_4_Action_"+str(action)+"_Events_MSA_Agent_"+str(self.agent))
        
        MSAREv_norm5 = tf.contrib.layers.layer_norm(inputs=MSAREv_l_hid_4, trainable=True, scope="NormLayer_5_Action_"+str(action)+"_Events_MSA_Agent_"+str(self.agent))
        MSAREv_out = tf.layers.dense(inputs=MSAREv_norm5, units=self.featsSAS_Events, activation=tf.nn.sigmoid, trainable=True, name="DenseLayer_5_Action_"+str(action)+"_Events_MSA_Agent_"+str(self.agent))

        print('MSAR_Out_Shape_A'+str(action), MSAR_out.get_shape())
        print('MSA_Events_Out_Shape_A'+str(action), MSAREv_out.get_shape())

        return (MSAR_out, MSAREv_out)

