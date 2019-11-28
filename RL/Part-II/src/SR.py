
# coding: utf-8

# In[ ]:


import numpy as np
import os
import tensorflow as tf
from Config import Config
from Env import Env
from NNPolicy import ANNPol
from NNPolicyWLN import ANNPolWLN
import time
from ReplicaNN import ReplicaNN
from ReplicaNNWLN import ReplicaNNWLN
from pathos.multiprocessing import ThreadingPool as Pool
import math
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# In[ ]:


class Stopwatch:

    # Construct self and start it running.
    def __init__(self):
        self._creationTime = time.time()  # Creation time

    # Return the elapsed time since creation of self, in seconds.
    def elapsedTime(self):
        return time.time() - self._creationTime


# In[ ]:


class NN:
    def __init__(self, inputLength, actionCount, input_var, agent, config, feats_S, feats_SAS, MInput_var, featsSAS_Events):
        
        # Config Instance
        self.config = config
        
        #Length of input state, comprises of (s) and (s')
        self.inputLength = inputLength
        
        # Number of features for encoder for \Phi(s) and \Phi(s')
        self.feats_S = feats_S
        
        # Number of features for encoder for \Phi(s,a,s')
        self.feats_SAS = feats_SAS
        
        # Number of features for events for \Phi(s,a,s')
        self.featsSAS_Events = featsSAS_Events
        
        # Number of actions, i.e. output units for m_sa network and \Phi(s,a,s') network
        self.act = actionCount
        
        # Placeholder for input, comprises of concatenated (s) and (s')
        self.input_var = input_var 
        
        # Placeholder for inputting Phi(s) as input to M network
        self.MInput_var = MInput_var
        
        # Which agent NN?
        self.agent = agent

        # Output tensor
        self.Encoder, self.Decoder, self.Phi_sas, self.weights, self.MSA, self.MSAEv = self.initNN()

    # Remember to change the mode while evaluating!
    def initNN(self):
        
        ################################
        # Encoder for states, s and s'
        # Input: (None,validLengthOfState)
        # Example: |s|:16*76, |s'|:16*76, feats_S=feats_SAS=30 --> |input_var|: 32*76, |output|: 32*30
        # s and s' concatenated
        # Single input represents one state only.
        # Output: (None, feats_S)
        ################################

        Enc_l_hid_1 = tf.layers.dense(inputs=self.input_var, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_1_State_Encoder_Agent_"+str(self.agent))

        Enc_l_hid_2 = tf.layers.dense(inputs=Enc_l_hid_1, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_2_State_Encoder_Agent_"+str(self.agent))

        Enc_l_hid_3 = tf.layers.dense(inputs=Enc_l_hid_2, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_3_State_Encoder_Agent_"+str(self.agent))

        Enc_l_hid_4 = tf.layers.dense(inputs=Enc_l_hid_3, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_4_State_Encoder_Agent_"+str(self.agent))

        Enc_out = tf.layers.dense(inputs=Enc_l_hid_4, units=self.feats_S, activation=None, trainable=True, name="DenseLayer_5_State_Encoder_Agent_"+str(self.agent))
        
        print "\n===Successfully initiated encoder neural network for agent "+str(self.agent)+"==="
        print('Encoder_Input_Shape', self.input_var.get_shape())
        print('Encoder_Out_Shape', Enc_out.get_shape())

        ################################
        # Decoder for states, s and s'
        # Input: (None, feats_S)
        # Example: |Phi_s|:16*30, |Phi_s'|:16*30, feats_S=feats_SAS=30 --> |input_var|: 32*30, |output|: 32*76
        # Phi_s and Phi_s' concatenated
        # Single input represents one Phi of state only.
        # Output: (None, validLengthOfState)        
        ################################
        
        Dec_l_hid_1 = tf.layers.dense(inputs=Enc_out, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_1_State_Decoder_Agent_"+str(self.agent))

        Dec_l_hid_2 = tf.layers.dense(inputs=Dec_l_hid_1, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_2_State_Decoder_Agent_"+str(self.agent))

        Dec_l_hid_3 = tf.layers.dense(inputs=Dec_l_hid_2, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_3_State_Decoder_Agent_"+str(self.agent))

        Dec_l_hid_4 = tf.layers.dense(inputs=Dec_l_hid_3, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_4_State_Decoder_Agent_"+str(self.agent))

        Dec_out = tf.layers.dense(inputs=Dec_l_hid_4, units=self.inputLength, activation=None, trainable=True, name="DenseLayer_5_State_Decoder_Agent_"+str(self.agent))
        
        print "\n===Successfully initiated decoder neural network for agent "+str(self.agent)+"==="
        print('Decoder_Input_Shape', Enc_out.get_shape())
        print('Decoder_Out_Shape', Dec_out.get_shape())
        
        ################################
        # Encoder for \Phi(s,a,s')
        # Input: (None, feats_S+feats_S)
        # Example: |Phi_s|:16*30, |Phi_s'|:16*30, |A|: 3, feats_S=feats_SAS=30 --> |input_var|: 16*60, |output|: 3*16*30
        # Phi_s and Phi_s' joined together
        # Single input represents one state and one destination state joined as one --> Phi(s,s').
        # Output: (NumberOfActions, None, feats_SAS)
        ################################
            
        Phi_out = []
        
        Enc_out_splitted = tf.split(Enc_out, 2, axis=0)
        EncSSD = tf.concat([Enc_out_splitted[0], Enc_out_splitted[1]], axis=1)

        Phi_l_hid_1 = tf.layers.dense(inputs=EncSSD, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_1_Phi_Encoder_Agent_"+str(self.agent))

        Phi_l_hid_2 = tf.layers.dense(inputs=Phi_l_hid_1, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_2_Phi_Encoder_Agent_"+str(self.agent))

        Phi_l_hid_3 = tf.layers.dense(inputs=Phi_l_hid_2, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_3_Phi_Encoder_Agent_"+str(self.agent))

        print "\n===Successfully initiated Phi_sas neural network for agent "+str(self.agent)+"==="
        print('Phi_Input_Shape', EncSSD.get_shape())

        for i in xrange(0, self.act):

            ################################
            # Action i
            ################################
            Phi_out.append(self.Phi_sasd_Network(Phi_l_hid_3, i))
            
        ################################
        # Reward Weights Network
        # Example: feats_S=feats_SAS=30 --> |output|: 30*1
        ################################
        
        reward_weights = tf.get_variable("Reward_Weights_Agent_"+str(self.agent), dtype=tf.float32, shape=[self.feats_SAS], trainable=True)        
        print "\n===Successfully initiated Reward Weight Variable for Agent "+str(self.agent)+"==="
        print('RW_Out_Shape', reward_weights.get_shape())
        
        ################################
        # M_SA network for estimating Q values using m_sa.w
        # Input: (None, feats_S)
        # Example: |Phi_s|:16*30, |Phi_s'|:16*30, |A|: 3, feats_S=feats_SAS=30 --> |input_var|: 32*30, |output|: 3*32*30
        # Takes in Phi_s
        # Single input represents one encoded state only.
        # Output: (NumberOfActions, None, feats_SAS)
        ################################        

        MSA_out = []
        MSA_Events = []

        MSA_l_hid_1 = tf.layers.dense(inputs=self.MInput_var, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_1_MSA_Agent_"+str(self.agent))

        MSA_l_hid_2 = tf.layers.dense(inputs=MSA_l_hid_1, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_2_MSA_Agent_"+str(self.agent))

        MSA_l_hid_3 = tf.layers.dense(inputs=MSA_l_hid_2, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_3_MSA_Agent_"+str(self.agent))

        print "\n===Successfully initiated M_sa neural network for agent "+str(self.agent)+"==="
        print('MSA_Input_Shape', self.MInput_var.get_shape())

        for i in xrange(0, self.act):

            ################################
            # Action i
            ################################
            
            MSAM, MSAE = self.M_sa_Network(MSA_l_hid_3, i)
            MSA_out.append(MSAM)
            MSA_Events.append(MSAE)

        return Enc_out, Dec_out, Phi_out, reward_weights, MSA_out, MSA_Events

    def Phi_sasd_Network(self, input_v, action):

        Phi_l_hid_4 = tf.layers.dense(inputs=input_v, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_4_Action_"+str(action)+"_Phi_Encoder_Agent_"+str(self.agent))
        Phi_out = tf.layers.dense(inputs=Phi_l_hid_4, units=self.feats_SAS, activation=None, trainable=True, name="DenseLayer_5_Action_"+str(action)+"_Phi_Encoder_Agent_"+str(self.agent))

        print('Phi_Out_Shape_A'+str(action), Phi_out.get_shape())

        return Phi_out

    def M_sa_Network(self, input_v, action):
        
        MSA_l_hid_4 = tf.layers.dense(inputs=input_v, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_4_Action_"+str(action)+"_MSA_Agent_"+str(self.agent))
        MSA_out = tf.layers.dense(inputs=MSA_l_hid_4, units=self.feats_SAS, activation=None, trainable=True, name="DenseLayer_5_Action_"+str(action)+"_MSA_Agent_"+str(self.agent))

        MSAEv_l_hid_4 = tf.layers.dense(inputs=input_v, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_4_Action_"+str(action)+"_Events_MSA_Agent_"+str(self.agent))
        MSAEv_out = tf.layers.dense(inputs=MSAEv_l_hid_4, units=self.featsSAS_Events, activation=tf.nn.sigmoid, trainable=True, name="DenseLayer_5_Action_"+str(action)+"_Events_MSA_Agent_"+str(self.agent))
        
        print('MSA_Out_Shape_A'+str(action), MSA_out.get_shape())
        print('MSA_Events_Out_Shape_A'+str(action), MSAEv_out.get_shape())
        
        return (MSA_out, MSAEv_out)
    
    def getEncodedState(self, flatobs, sess):
        return self.Encoder.eval(session=sess, feed_dict={self.input_var: flatobs})


# In[ ]:


class NNWLN:
    def __init__(self, inputLength, actionCount, input_var, agent, config, feats_S, feats_SAS, MInput_var, featsSAS_Events):
        
        # Config Instance
        self.config = config
        
        #Length of input state, comprises of (s) and (s')
        self.inputLength = inputLength
        
        # Number of features for encoder for \Phi(s) and \Phi(s')
        self.feats_S = feats_S
        
        # Number of features for encoder for \Phi(s,a,s')
        self.feats_SAS = feats_SAS
        
        # Number of features for events for \Phi(s,a,s')
        self.featsSAS_Events = featsSAS_Events
        
        # Number of actions, i.e. output units for m_sa network and \Phi(s,a,s') network
        self.act = actionCount
        
        # Placeholder for input, comprises of concatenated (s) and (s')
        self.input_var = input_var 
        
        # Placeholder for inputting Phi(s) as input to M network
        self.MInput_var = MInput_var
        
        # Which agent NN?
        self.agent = agent

        # Output tensor
        self.Encoder, self.Decoder, self.Phi_sas, self.weights, self.MSA, self.MSAEv = self.initNN()

    # Remember to change the mode while evaluating!
    def initNN(self):
        
        ################################
        # Encoder for states, s and s'
        # Input: (None,validLengthOfState)
        # Example: |s|:16*76, |s'|:16*76, feats_S=feats_SAS=30 --> |input_var|: 32*76, |output|: 32*30
        # s and s' concatenated
        # Single input represents one state only.
        # Output: (None, feats_S)
        ################################

        Enc_norm1 = tf.contrib.layers.layer_norm(inputs=self.input_var, trainable=True, scope="NormLayer_1_State_Encoder_Agent_"+str(self.agent))
        Enc_l_hid_1 = tf.layers.dense(inputs=Enc_norm1, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_1_State_Encoder_Agent_"+str(self.agent))

        Enc_norm2 = tf.contrib.layers.layer_norm(inputs=Enc_l_hid_1, trainable=True, scope="NormLayer_2_State_Encoder_Agent_"+str(self.agent))
        Enc_l_hid_2 = tf.layers.dense(inputs=Enc_norm2, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_2_State_Encoder_Agent_"+str(self.agent))

        Enc_norm3 = tf.contrib.layers.layer_norm(inputs=Enc_l_hid_2, trainable=True, scope="NormLayer_3_State_Encoder_Agent_"+str(self.agent))
        Enc_l_hid_3 = tf.layers.dense(inputs=Enc_norm3, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_3_State_Encoder_Agent_"+str(self.agent))

        Enc_norm4 = tf.contrib.layers.layer_norm(inputs=Enc_l_hid_3, trainable=True, scope="NormLayer_4_State_Encoder_Agent_"+str(self.agent))
        Enc_l_hid_4 = tf.layers.dense(inputs=Enc_norm4, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_4_State_Encoder_Agent_"+str(self.agent))

        Enc_norm5 = tf.contrib.layers.layer_norm(inputs=Enc_l_hid_4, trainable=True, scope="NormLayer_5_State_Encoder_Agent_"+str(self.agent))
        Enc_out = tf.layers.dense(inputs=Enc_norm5, units=self.feats_S, activation=None, trainable=True, name="DenseLayer_5_State_Encoder_Agent_"+str(self.agent))
        
        print "\n===Successfully initiated encoder neural network for agent "+str(self.agent)+"==="
        print('Encoder_Input_Shape', self.input_var.get_shape())
        print('Encoder_Out_Shape', Enc_out.get_shape())

        ################################
        # Decoder for states, s and s'
        # Input: (None, feats_S)
        # Example: |Phi_s|:16*30, |Phi_s'|:16*30, feats_S=feats_SAS=30 --> |input_var|: 32*30, |output|: 32*76
        # Phi_s and Phi_s' concatenated
        # Single input represents one Phi of state only.
        # Output: (None, validLengthOfState)        
        ################################
        
        Dec_norm1 = tf.contrib.layers.layer_norm(inputs=Enc_out, trainable=True, scope="NormLayer_1_State_Decoder_Agent_"+str(self.agent))
        Dec_l_hid_1 = tf.layers.dense(inputs=Dec_norm1, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_1_State_Decoder_Agent_"+str(self.agent))

        Dec_norm2 = tf.contrib.layers.layer_norm(inputs=Dec_l_hid_1, trainable=True, scope="NormLayer_2_State_Decoder_Agent_"+str(self.agent))
        Dec_l_hid_2 = tf.layers.dense(inputs=Dec_norm2, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_2_State_Decoder_Agent_"+str(self.agent))

        Dec_norm3 = tf.contrib.layers.layer_norm(inputs=Dec_l_hid_2, trainable=True, scope="NormLayer_3_State_Decoder_Agent_"+str(self.agent))
        Dec_l_hid_3 = tf.layers.dense(inputs=Dec_norm3, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_3_State_Decoder_Agent_"+str(self.agent))

        Dec_norm4 = tf.contrib.layers.layer_norm(inputs=Dec_l_hid_3, trainable=True, scope="NormLayer_4_State_Decoder_Agent_"+str(self.agent))
        Dec_l_hid_4 = tf.layers.dense(inputs=Dec_norm4, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_4_State_Decoder_Agent_"+str(self.agent))

        Dec_norm5 = tf.contrib.layers.layer_norm(inputs=Dec_l_hid_4, trainable=True, scope="NormLayer_5_State_Decoder_Agent_"+str(self.agent))
        Dec_out = tf.layers.dense(inputs=Dec_norm5, units=self.inputLength, activation=None, trainable=True, name="DenseLayer_5_State_Decoder_Agent_"+str(self.agent))
        
        print "\n===Successfully initiated decoder neural network for agent "+str(self.agent)+"==="
        print('Decoder_Input_Shape', Enc_out.get_shape())
        print('Decoder_Out_Shape', Dec_out.get_shape())
        
        ################################
        # Encoder for \Phi(s,a,s')
        # Input: (None, feats_S+feats_S)
        # Example: |Phi_s|:16*30, |Phi_s'|:16*30, |A|: 3, feats_S=feats_SAS=30 --> |input_var|: 16*60, |output|: 3*16*30
        # Phi_s and Phi_s' joined together
        # Single input represents one state and one destination state joined as one --> Phi(s,s').
        # Output: (NumberOfActions, None, feats_SAS)
        ################################
            
        Phi_out = []
        
        Enc_out_splitted = tf.split(Enc_out, 2, axis=0)
        EncSSD = tf.concat([Enc_out_splitted[0], Enc_out_splitted[1]], axis=1)

        Phi_norm1 = tf.contrib.layers.layer_norm(inputs=EncSSD, trainable=True, scope="NormLayer_1_Phi_Encoder_Agent_"+str(self.agent))
        Phi_l_hid_1 = tf.layers.dense(inputs=Phi_norm1, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_1_Phi_Encoder_Agent_"+str(self.agent))

        Phi_norm2 = tf.contrib.layers.layer_norm(inputs=Phi_l_hid_1, trainable=True, scope="NormLayer_2_Phi_Encoder_Agent_"+str(self.agent))
        Phi_l_hid_2 = tf.layers.dense(inputs=Phi_norm2, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_2_Phi_Encoder_Agent_"+str(self.agent))

        Phi_norm3 = tf.contrib.layers.layer_norm(inputs=Phi_l_hid_2, trainable=True, scope="NormLayer_3_Phi_Encoder_Agent_"+str(self.agent))
        Phi_l_hid_3 = tf.layers.dense(inputs=Phi_norm3, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_3_Phi_Encoder_Agent_"+str(self.agent))

        print "\n===Successfully initiated Phi_sas neural network for agent "+str(self.agent)+"==="
        print('Phi_Input_Shape', EncSSD.get_shape())

        for i in xrange(0, self.act):

            ################################
            # Action i
            ################################
            Phi_out.append(self.Phi_sasd_Network(Phi_l_hid_3, i))
            
        ################################
        # Reward Weights Network
        # Example: feats_S=feats_SAS=30 --> |output|: 30*1
        ################################
        
        reward_weights = tf.get_variable("Reward_Weights_Agent_"+str(self.agent), dtype=tf.float32, shape=[self.feats_SAS], trainable=True)        
        print "\n===Successfully initiated Reward Weight Variable for Agent "+str(self.agent)+"==="
        print('RW_Out_Shape', reward_weights.get_shape())
        
        ################################
        # M_SA network for estimating Q values using m_sa.w
        # Input: (None, feats_S)
        # Example: |Phi_s|:16*30, |Phi_s'|:16*30, |A|: 3, feats_S=feats_SAS=30 --> |input_var|: 32*30, |output|: 3*32*30
        # Takes in Phi_s
        # Single input represents one encoded state only.
        # Output: (NumberOfActions, None, feats_SAS)
        ################################        

        MSA_out = []
        MSA_Events = []

        MSA_norm1 = tf.contrib.layers.layer_norm(inputs=self.MInput_var, trainable=True, scope="NormLayer_1_MSA_Agent_"+str(self.agent))
        MSA_l_hid_1 = tf.layers.dense(inputs=MSA_norm1, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_1_MSA_Agent_"+str(self.agent))

        MSA_norm2 = tf.contrib.layers.layer_norm(inputs=MSA_l_hid_1, trainable=True, scope="NormLayer_2_MSA_Agent_"+str(self.agent))
        MSA_l_hid_2 = tf.layers.dense(inputs=MSA_norm2, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_2_MSA_Agent_"+str(self.agent))

        MSA_norm3 = tf.contrib.layers.layer_norm(inputs=MSA_l_hid_2, trainable=True, scope="NormLayer_3_MSA_Agent_"+str(self.agent))
        MSA_l_hid_3 = tf.layers.dense(inputs=MSA_norm3, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_3_MSA_Agent_"+str(self.agent))

        print "\n===Successfully initiated M_sa neural network for agent "+str(self.agent)+"==="
        print('MSA_Input_Shape', self.MInput_var.get_shape())

        for i in xrange(0, self.act):

            ################################
            # Action i
            ################################
            
            MSAM, MSAE = self.M_sa_Network(MSA_l_hid_3, i)
            MSA_out.append(MSAM)
            MSA_Events.append(MSAE)

        return Enc_out, Dec_out, Phi_out, reward_weights, MSA_out, MSA_Events

    def Phi_sasd_Network(self, input_v, action):

        Phi_norm4 = tf.contrib.layers.layer_norm(inputs=input_v, trainable=True, scope="NormLayer_4_Action_"+str(action)+"_Phi_Encoder_Agent_"+str(self.agent))
        Phi_l_hid_4 = tf.layers.dense(inputs=Phi_norm4, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_4_Action_"+str(action)+"_Phi_Encoder_Agent_"+str(self.agent))

        Phi_norm5 = tf.contrib.layers.layer_norm(inputs=Phi_l_hid_4, trainable=True, scope="NormLayer_5_Action_"+str(action)+"_Phi_Encoder_Agent_"+str(self.agent))
        Phi_out = tf.layers.dense(inputs=Phi_norm5, units=self.feats_SAS, activation=None, trainable=True, name="DenseLayer_5_Action_"+str(action)+"_Phi_Encoder_Agent_"+str(self.agent))

        print('Phi_Out_Shape_A'+str(action), Phi_out.get_shape())

        return Phi_out

    def M_sa_Network(self, input_v, action):
        
        MSA_norm4 = tf.contrib.layers.layer_norm(inputs=input_v, trainable=True, scope="NormLayer_4_Action_"+str(action)+"_MSA_Agent_"+str(self.agent))
        MSA_l_hid_4 = tf.layers.dense(inputs=MSA_norm4, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_4_Action_"+str(action)+"_MSA_Agent_"+str(self.agent))

        MSA_norm5 = tf.contrib.layers.layer_norm(inputs=MSA_l_hid_4, trainable=True, scope="NormLayer_5_Action_"+str(action)+"_MSA_Agent_"+str(self.agent))
        MSA_out = tf.layers.dense(inputs=MSA_norm5, units=self.feats_SAS, activation=None, trainable=True, name="DenseLayer_5_Action_"+str(action)+"_MSA_Agent_"+str(self.agent))

        MSAEv_norm4 = tf.contrib.layers.layer_norm(inputs=input_v, trainable=True, scope="NormLayer_4_Action_"+str(action)+"_Events_MSA_Agent_"+str(self.agent))
        MSAEv_l_hid_4 = tf.layers.dense(inputs=MSAEv_norm4, units=self.config.numUnitsPerLayer, activation=tf.nn.relu, trainable=True, name="DenseReLuLayer_4_Action_"+str(action)+"_Events_MSA_Agent_"+str(self.agent))
        
        MSAEv_norm5 = tf.contrib.layers.layer_norm(inputs=MSAEv_l_hid_4, trainable=True, scope="NormLayer_5_Action_"+str(action)+"_Events_MSA_Agent_"+str(self.agent))
        MSAEv_out = tf.layers.dense(inputs=MSAEv_norm5, units=self.featsSAS_Events, activation=tf.nn.sigmoid, trainable=True, name="DenseLayer_5_Action_"+str(action)+"_Events_MSA_Agent_"+str(self.agent))
        
        print('MSA_Out_Shape_A'+str(action), MSA_out.get_shape())
        print('MSA_Events_Out_Shape_A'+str(action), MSAEv_out.get_shape())
        
        return (MSA_out, MSAEv_out)
    
    def getEncodedState(self, flatobs, sess):
        return self.Encoder.eval(session=sess, feed_dict={self.input_var: flatobs})


# In[ ]:


class Train():
    
    def __init__(self, agent, config):
        
        # Configuration file object.
        self.config = config
        
        # Line of agent.
        self.line = self.config.agentWiseLines[agent]
        
        # All locations (shared/private) assigned to agent.
        self.locations = self.line.getLocations()
#         print self.locations
        
        # Create silmulator for the agent.
        self.env_ = Env(agent, self.config)
        
        # Training class for the corresponding agent.
        self.agent = agent
        
        # Batch size to be used for training.
        self.batch = self.config.batch
        
        # Policy NN instance
        self.policyNN = None
        
        # NN Instance
        self.NNInstance = None
        
        # Number of features for Phi_s
        self.featsS = self.config.featsS
        
        # Number of features for Phi_sas
        self.featsSAS = self.config.featsSAS
        
        self.eventMap = dict()
        self.reset_times = []
        self.featsSAS_Events = None

        self.resetMappingFunction(updateResetTimes=True)
        self.constraints_rewards = self.constraintRewardsVector()
        self.numberOfConstraints = len(self.constraints_rewards)
        
        self.deli = int(float(self.env_.T) / float(self.env_.actionTime))

#         print self.constraints_rewards
#         print self.reset_times

        self.finalPATHS = None
        self.testMap = None
            
    ''' Function to convert action id to 1-hot representation'''
    def to_one_hot(self, id, N):
        arr = np.zeros(shape=(N), dtype=float)
        arr[id] = 1
        return arr
    
    def constraintRewardsVector(self):
        constraints_rewards = []
        for j in xrange(0, len(self.config.sharedSites)):
            locs = self.config.sharedSites[j]
            if self.line.hasLocation(locs):
                constraints_rewards.extend([self.config.creward[j]]*len(self.reset_times))
        return constraints_rewards
    
    def computeProdProbability(self, event_maps, MVEventsAllAgents):
        
        testBitNumber = 0
        products = []
        for j in xrange(0, len(self.config.sharedSites)):
            owners = []
            locs = self.config.sharedSites[j]

            if not self.line.hasLocation(locs):
                continue
                
            for l in self.config.lines:
                if l.hasLocation(locs):
                    owners.extend(l.getOwners())
#             print owners
                
            for k2 in sorted(self.eventMap[locs].iterkeys(), reverse=True):
#                 print sorted(self.eventMap[locs].iterkeys(), reverse=True)
#                 print testBitNumber, " ", self.eventMap[locs][k2]["bit"], " ", locs, " ", k2
                assert self.eventMap[locs][k2]["bit"] == testBitNumber

                # Each event as bit number verified
#                 print "Generating product with location ", locs, " and Time range ", k2, " - ", k2 - self.config.repeatKTimeSteps
                prod = 1.0

                for own in owners:
                    
                    if (own == self.agent):
                        continue
                    
                    which_bit = event_maps[own][locs][k2]["bit"]
                    extracting_bit = MVEventsAllAgents[own][:,which_bit]
                    extracting_bit = np.array(extracting_bit)
                    prod *= (1.0 - extracting_bit)
                products.append(prod)

                testBitNumber += 1
        products = np.array(products)
        expected_product = np.mean(products, axis=1)
        return expected_product
        
    def resetMappingFunction(self, updateResetTimes = False):

        self.eventMap["agent"] = self.agent
        self.featsSAS_Events = 0
        
        for locs in self.line.getSharedLocations():

            loc = locs
            self.eventMap[loc] = dict()

            T = self.config.T
            action = self.config.T

            for start in xrange(T, 0, -self.config.repeatKTimeSteps):
                if(start - self.config.actionTimes[self.agent] < 0):
                    continue

                if updateResetTimes:
                    self.reset_times.append(start)

                self.eventMap[loc][start] = dict()
                self.eventMap[loc][start]["val"] = 0
                self.eventMap[loc][start]["bit"] = self.featsSAS_Events
                self.featsSAS_Events += 1

                if (start - self.config.repeatKTimeSteps >= 0):
                    end = start - self.config.repeatKTimeSteps
                else:
                    end = 0

                while(action > end):
                    self.eventMap[loc][start][action] = dict()
#                     self.eventMap[loc][start][action]["sample"] = []
#                     self.eventMap[loc][start][action]["decision"] = -1

                    action -= self.config.actionTimes[self.agent]

            updateResetTimes = False
#         print json.dumps(self.eventMap, indent=6, sort_keys=True)

    def getDataForObs(self, obs):
        location_obs = self.env_.getLocation(obs)
        time_obs = self.env_.getTime(obs)
        inspection_obs = self.env_.getInspectionBits(obs)
        dold_obs = self.env_.getOld(obs)
        actualLocation = self.locations[location_obs]

        return (location_obs, time_obs, inspection_obs, dold_obs, actualLocation)

    def getTimeSlot(self, time):
        assert len(self.reset_times) != 0
        for i in xrange(1, len(self.reset_times)):
            start = self.reset_times[i-1]
            end = self.reset_times[i]

            if (time <= start and time > end):
                return start
        return self.reset_times[-1]

    def indicatorFunctionForEvents(self, state, action, statedash, batch):
        
        assert self.testMap != None
        
#         print self.env_.printState(state)
#         print action
#         print self.env_.printState(statedash)

        returnBits = [0.0]*self.featsSAS_Events
        # Action should be 0 -> Inspect action for any event.
        if action != 0:
            return returnBits
        
        (location_obs, time_obs, inspection_obs, dold_obs, actualLocation_obs) = self.getDataForObs(state)
        (location_nextobs, time_nextobs, inspection_nextobs, dold_nextobs, actualLocation_nextobs) = self.getDataForObs(statedash)
        
        # Only if it is a shared location!
        if action == 0 and actualLocation_obs == actualLocation_nextobs and location_obs == location_nextobs and self.line.isShared(actualLocation_obs):
            if (inspection_obs[location_obs] == 0 and dold_obs == 0):
                if (inspection_nextobs[location_nextobs] == 1 and dold_nextobs == 0):
                    returnBits[self.eventMap[actualLocation_obs][self.getTimeSlot(time_obs)]["bit"]] = 1.0
                    self.testMap[batch][actualLocation_obs][self.getTimeSlot(time_obs)] = True
                
        return returnBits

    def runIteration(self, sess):

        # store sampled states
        observations = []

        # store sampled actions i
        actions = []

        # Empirical return for each sampled state-action pair
        rewards = []

        # Keep track of the states sampled after the current state
        nextObs = []

        # Keep track of rewards for all actions from a particular start state and end state.
        all_rewards = []
        
        s = self.env_.reset()
        
        numberOfSampledActions = int(float(self.env_.T) / float(self.env_.actionTime))
        
        episodeEnd = False
                
        '''sample "mini-batch" trajectories'''
        while (episodeEnd != True):

            s = np.array(s).reshape((self.batch, self.env_.validLengthOfState))

            randomOrPolicy = np.random.choice(2, 1, p=[self.config.randomActionProb, 1-self.config.randomActionProb])
            if randomOrPolicy==0:
                a = [np.random.choice(self.env_.numberOfActions, 1) for i in xrange(self.batch)]
            elif randomOrPolicy==1:
                #enc_s = self.NNInstance.getEncodedState(neural_state_s, sess)
                probs = self.policyNN.l_prob.eval({self.policyNN.input_var: s, self.policyNN.train_mode: False}, session=sess)
                a = [np.random.choice(self.env_.numberOfActions, 1, p=probs[i]) for i in xrange(self.batch)]

            (s_, rew_, episodeEnd) = self.env_.step(a)

            if (episodeEnd):
                break

            s_ = np.array(s_).reshape((self.batch, self.env_.validLengthOfState))

            '''Append all data to lists'''
            # append current state
            observations.append(s)            
            # append current action
            one_hot_acts = [self.to_one_hot(a[i], self.env_.numberOfActions) for i in xrange(self.batch)]
            actions.append(one_hot_acts) 
            # append immed reward received
            rewards.append(rew_)       
            # append subsequent state to list
            nextObs.append(s_)
            
            s = s_
    
        observations = np.array(observations)
        observations = [observations[:,i] for i in xrange(self.batch)]
#         print np.array(observations).shape
        
        actions = np.array(actions)
        actions = [actions[:,i] for i in xrange(self.batch)]
#         print np.array(actions).shape

        rewards = np.array(rewards)
        rewards = np.transpose(rewards)
#         print rewards.shape

        nextObs = np.array(nextObs)
        nextObs = [nextObs[:,i] for i in xrange(self.batch)]
#         print np.array(nextObs).shape

        self.testMap = self.resetTestMappingFunction()
        
        all_rewards_ssd = []
        all_indicators_ssd = []
        for b in xrange(self.batch):
            all_rew = []
            all_ind = []
            for th in xrange(numberOfSampledActions):
                all_rew_a = []
                s  = np.array(observations[b][th]).tolist()
                s_ = np.array(nextObs[b][th]).tolist()
                for act in xrange(self.env_.numberOfActions):
                    all_rew_a.append(self.env_.rewardFunction(s, act, s_))
                samp_acti_ = self.env_.getAction(actions[b][th])
                all_ind.append(self.indicatorFunctionForEvents(s, samp_acti_, s_, b))
                all_rew.append(all_rew_a)
            all_rewards_ssd.append(all_rew)
            all_indicators_ssd.append(all_ind)
        all_rewards_ssd = np.array(all_rewards_ssd)
        all_indicators_ssd = np.array(all_indicators_ssd)
        
#         print json.dumps(self.testMap, indent=4, sort_keys=True)
        
        returns = []
        rewards = rewards.tolist()
        for b in xrange(self.batch):
            ret = []
            return_so_far = 0
            for t in range(len(rewards[b]) - 1, -1, -1):
                return_so_far = (rewards[b][t] + return_so_far)
                ret.append(return_so_far)
            ret = ret[::-1]
            returns.append(ret)
        rewards = np.array(rewards)
        returns = np.array(returns)

#             nextactions = actions[1:]
#             nextactions.append([0]*self.env_.numberOfActions)
        
        # collect all states in an array with dimension N*S
        obs = np.concatenate(observations, axis=0)
        obs = np.array(obs).reshape((self.batch * int(float(self.env_.T) / float(self.env_.actionTime)), self.env_.validLengthOfState))
#         print obs.shape
        
        # collect all actions, dimension N*A
        act = np.concatenate(actions, axis=0)
#         print act.shape

#         nextacts = np.concatenate([p["nextactions"] for p in PATHS])
        
        # Used for extracting values according to samples actions from tensors of all actions.
        extractor = []
        numOfSampledActions = act.shape[0]
        for i in xrange(0, numOfSampledActions):
            extractor.append([self.env_.getAction(act[i]), i])
        extractor = np.array(extractor)
#         print extractor.shape

        deli = int(float(self.env_.T) / float(self.env_.actionTime))
        start_state_extractor = []
#         startacts = []
        for i in xrange(0, self.batch*deli, deli):
            start_state_extractor.append([i])
#             startacts.append(act[i])
        start_state_extractor = np.array(start_state_extractor)
#         startacts = np.array(startacts)
#         print startacts, startacts.shape
#         print start_state_extractor
        
        # collect all next states in an array with dimension N*S
        nextobs = np.concatenate(nextObs, axis=0)
        nextobs = np.array(nextobs).reshape((self.batch * int(float(self.env_.T) / float(self.env_.actionTime)), self.env_.validLengthOfState))
#         print nextobs.shape
        
        # Next action probabilities
        feeding_policy = dict()
#        feeding_policy[self.NNInstance.input_var] = nextobs
#        encodedSDash = np.array(sess.run(self.NNInstance.Encoder, feed_dict=feeding_policy))
        
#        feeding_policy[self.NNInstance.input_var] = obs
#        encodedS = np.array(sess.run(self.NNInstance.Encoder, feed_dict=feeding_policy))
        
        feeding_policy[self.policyNN.input_var] = nextobs
        feeding_policy[self.policyNN.train_mode] = False

        # Pi(a'|s') for all actions a', fixing Pi(a'|s_H) = 0.0 for all actions
        deli = int(float(self.env_.T) / float(self.env_.actionTime))
        next_act_prob_sd = self.policyNN.l_prob.eval(session=sess, feed_dict=feeding_policy)
        next_act_prob_sd = np.array(next_act_prob_sd).reshape((self.batch * int(float(self.env_.T) / float(self.env_.actionTime)), self.env_.numberOfActions))
        for x in range(deli-1, len(next_act_prob_sd), deli):
            next_act_prob_sd[x] = [0.0]*self.env_.numberOfActions
        next_act_prob_sd = np.array(next_act_prob_sd)
#         print next_act_prob_sd.shape
            
        feeding_policy = dict()
        feeding_policy[self.policyNN.input_var] = obs
        feeding_policy[self.policyNN.train_mode] = False
        
        # Pi(a|s) for all actions a
        next_act_prob_s = self.policyNN.l_prob.eval(session=sess, feed_dict=feeding_policy)
        next_act_prob_s = np.array(next_act_prob_s).reshape((self.batch * int(float(self.env_.T) / float(self.env_.actionTime)), self.env_.numberOfActions))
#         print next_act_prob_s.shape
        
        # Collect all rewards
        rew = np.concatenate(rewards, axis=0)
#         print rew.shape
        
        # all returns, single dimension vector (N,)
        ret = np.concatenate(returns, axis=0)   
#         print ret.shape

        # Concat s and s' for input to NN.
        obs_nextobs_appended = np.append(obs, nextobs, axis=0)
#         print obs_nextobs_appended.shape

        # Collect all, all rewards
        all_rews = np.concatenate(all_rewards_ssd, axis=0)
        all_rews = np.array(np.transpose(all_rews))
#         print all_rews.shape

        sampled_indicators = np.concatenate(all_indicators_ssd, axis=0)
#         print all_indicators_ssd.shape
#         print sampled_indicators.shape

        sampled_indicators_prods = []

        for bh in xrange(0, self.batch*deli, deli):
            extracted_ind = sampled_indicators[bh:bh+deli-1]
            extracted_ind = np.array(extracted_ind)
            extracted_ind1M = 1.0 - extracted_ind
            batch_prod = []
            batch_prod.append(extracted_ind1M[0])
            for tt in xrange(1, extracted_ind1M.shape[0]):
                pro = batch_prod[tt-1]*extracted_ind1M[tt]
                batch_prod.append(pro)
            batch_prod = np.array(batch_prod)
            sampled_indicators_prods.append(batch_prod)
        sampled_indicators_prods = np.array(sampled_indicators_prods)                
    
        avgMDPReturn = np.mean([returns[i][0] for i in xrange(self.batch)])

        self.finalPATHS = dict(
            observations = obs,
            obs_nextobs = obs_nextobs_appended,
            sampled_actions = act,
            all_act_rewards = all_rews,
            extractor_vals = extractor,
            nextActionProbsSD = next_act_prob_sd,
            nextobservations = nextobs,
            sampled_act_indicators = sampled_indicators,
            start_st_extractor = start_state_extractor,
            #sampled_next_acts = nextacts,
            # To be used by evaluator only.
            nextActionProbsS = next_act_prob_s,
            avgMDPValue = avgMDPReturn,
            indicatorProducts = sampled_indicators_prods
            #startactions = startacts
        )

        return self

    def giveMeEverything(self, p):
        
        assert self.finalPATHS != None
        
        feeding = dict()
        feeding[p.input_var] = self.finalPATHS["obs_nextobs"]
        feeding[p.act_rew] = self.finalPATHS["all_act_rewards"]
        feeding[p.Mextract] = self.finalPATHS["extractor_vals"]
        feeding[p.actProbSD] = self.finalPATHS["nextActionProbsSD"]
        feeding[p.policy.input_var] = self.finalPATHS["observations"]
        feeding[p.policy.train_mode] = False
        feeding[p.Term1EventsPlaceholder] = self.finalPATHS["sampled_act_indicators"]
        feeding[p.startStateExtractor] = self.finalPATHS["start_st_extractor"]
#         feeding[p.startactions] = self.finalPATHS["startactions"]
        feeding[p.sampledActs] = self.finalPATHS["sampled_actions"]
        feeding[p.indicatorProducts] = self.finalPATHS["indicatorProducts"]

        return feeding
    
    def resetTestMappingFunction(self):

        testMap = dict()
        testMap["agent"] = self.agent

        for batch in xrange(0, self.batch):

            testMap[batch] = dict()
            for locs in self.line.getSharedLocations():

                loc = locs
                testMap[batch][loc] = dict()

                T = self.config.T

                for start in xrange(T, 0, -self.config.repeatKTimeSteps):
                    if (start - self.config.actionTimes[self.agent] < 0):
                        continue

                    testMap[batch][loc][start] = False

#         print json.dumps(testMap, indent=4, sort_keys=True)
        return testMap

    



# In[ ]:


class PlaceholderClass():
    
    """
        t -> Train Instance
        c -> Config Instance
        agent -> Which agent?!
    """
    def __init__(self, t, c, agent):
        
        # Train Instance
        self.t = t
    
        # Config Instance
        self.c = c
        
        # Which agent?
        self.agent = agent
        
        # Number of features for Phi_s
        self.featsS = self.t.featsS
        
        # Number of features for Phi_sas
        self.featsSAS = self.t.featsSAS
        
        # Number of features for events for Phi_sas
        self.featsSAS_Events = self.t.featsSAS_Events
        
        # Input placeholder
        self.input_var  = tf.placeholder(shape=[None, self.t.env_.validLengthOfState], dtype=tf.float32, name="input_var_"+str(agent))
        
        # Input for M network placeholder
        self.MInput_var = tf.placeholder(shape=[None, self.t.env_.validLengthOfState], dtype=tf.float32)
        
        # initialize NN
        if self.c.NNlayerNorm:
            print "Defining Enc/Dec/Phi/M NN with LN"
            self.NN = NNWLN(inputLength=self.t.env_.validLengthOfState, actionCount=t.env_.numberOfActions, input_var=self.input_var, agent=agent, config=c, feats_S=self.featsS, feats_SAS=self.featsSAS, MInput_var=self.MInput_var, featsSAS_Events=self.featsSAS_Events)
        else:
            print "Defining Enc/Dec/Phi/M NN without LN"
            self.NN = NN(inputLength=self.t.env_.validLengthOfState, actionCount=t.env_.numberOfActions, input_var=self.input_var, agent=agent, config=c, feats_S=self.featsS, feats_SAS=self.featsSAS, MInput_var=self.MInput_var, featsSAS_Events=self.featsSAS_Events)
        
        # Rewards input
        self.act_rew = tf.placeholder(shape=[self.t.env_.numberOfActions, (self.t.batch * int(float(self.t.env_.T) / float(self.t.env_.actionTime)))], dtype=tf.float32)

        # Input variable for policy
        self.policy_input = tf.placeholder(shape=[None, self.t.env_.validLengthOfState], dtype=tf.float32, name="policy_input_var_"+str(agent))
        
        # Train mode for policy NNpolicy_input
        self.policy_train_mode = tf.placeholder(dtype=tf.bool)
        
        # Learned Policy Instance
        if self.c.PollayerNorm:
            print "Defining Policy NN with LN"
            self.policy = ANNPolWLN(self.t.env_.validLengthOfState, self.t.env_.numberOfActions, self.policy_input, self.policy_train_mode, agent, c)            
        else:
            print "Defining Policy NN without LN"
            self.policy = ANNPol(self.t.env_.validLengthOfState, self.t.env_.numberOfActions, self.policy_input, self.policy_train_mode, agent, c)
        
        # Extractor for M values according to sampled actions from array of MSA values for all actions
        self.Mextract = tf.placeholder(shape=[None, 2], dtype=tf.int32)
        
        # Placeholder for next observations
        self.next_observations = tf.placeholder(shape=[None, self.t.env_.validLengthOfState], dtype=tf.float32)
        
        self.encoded_out = None
        self.decoded_out = None
        self.Phi_out = None
        self.r_weight = None
        self.MSA_out= None
        self.state_enc_dec_loss = None
        
        self.predicted_reward = None
        self.rew_loss = None
        self.loss1 = None        
        self.learning_rate1 = None
        self.optimizer1 = None
        self.learning_step1 = None

        self.Term1Placeholder = None
        self.PhiSASAllActions = None
        self.PhiSAS_sampled_act = None
        self.oldMSDV = None
        self.MSA_sampled_act = None
        self.actProb = None
        self.actProb_splitted = None
        self.prodActoldMSDV = None
        self.sumOfProds = None
        self.encoded_out_splitted = None
        self.PhiS = None
        self.PhiSD = None

        self.loss2 = None        
        self.learning_rate2 = None
        self.optimizer2 = None
        self.learning_step2 = None
        
    def computationalGraphs(self):

        # -----------------------------------------------------------------------------------------------------
        # NN Computational Graph
        # -----------------------------------------------------------------------------------------------------
        # Shape = None 
        self.encoded_out = self.NN.Encoder

        self.decoded_out = self.NN.Decoder
        
        self.Phi_out = self.NN.Phi_sas
        
        self.r_weight = self.NN.weights
        
        self.MSA_out = self.NN.MSA
        
        self.MSA_Events = self.NN.MSAEv
        
        #######################
        # Utilities for Term 2, Loss 2
        #######################
        
        # Encoded concatenated s and s', splitted into s and s' separately
        self.encoded_out_splitted = tf.split(self.encoded_out, 2, axis=0)
        
        # Encoded s only
        self.PhiS = self.encoded_out_splitted[0]

        # Encoded s' only
        self.PhiSD = self.encoded_out_splitted[1]
        
        ##############################
        # Encoder Decoder Loss
        ##############################

        self.state_enc_dec_loss = tf.losses.mean_squared_error(predictions=self.decoded_out, labels=self.input_var)
        
        ##############################
        # Loss for Phi(s,a,s') and R(s,a,s') learning w
        ##############################

        # Multiplies 3*N*30 Phi_out to (30,) weight vector, takes sum across axis=2 for dot product and flattens the array.
        self.predicted_reward = tf.reduce_sum(tf.multiply(self.r_weight, self.Phi_out), axis=2)
        self.rew_loss = tf.losses.mean_squared_error(predictions=self.predicted_reward, labels=self.act_rew)
        
        ##############################
        # Total Loss for Encoder/Decoder and Rewards
        ##############################

        self.loss1 = self.state_enc_dec_loss + self.rew_loss

        # Learning Rate
        self.learning_rate1 = self.c.EDlearningRate
        
        # Defining Optimizer
        self.optimizer1 = tf.train.AdamOptimizer(self.learning_rate1)
        
        # Update Gradients
        self.learning_step1 = (self.optimizer1.minimize(self.loss1))

        ##############################
        # Loss for learning Q/M values
        ##############################
        
        #######################
        # Term 1
        #######################
        
        # Phi(s,a,s') signal for Loss 2
        self.Term1Placeholder = tf.placeholder(shape=[None, self.featsSAS], dtype=tf.float32)

        #######################
        # Utlities - Term 1
        #######################        
        
        self.sampledActs = tf.placeholder(shape=[None, self.t.env_.numberOfActions], dtype=tf.float32)
        
#         self.sampledActs_spl = tf.split(self.sampledActs, self.t.env_.numberOfActions, axis=1)
        
        # Phi(s,a,s') for all actions, sampled s,s'
        self.PhiSASAllActions = self.Phi_out
        
        #self.prodSampledActsPhiSASAll = [self.sampledActs_spl[act]*self.PhiSASAllActions[act] for act in xrange(0, self.t.env_.numberOfActions)]
        
        #self.PhiSAS_sampled_act = tf.reduce_sum(self.prodSampledActsPhiSASAll, axis=0)

        # Sampled Phi(s,a,s') acc to sampled actions. [Term 1 of loss function]
        self.PhiSAS_sampled_act = tf.gather_nd(indices=self.Mextract, params=self.PhiSASAllActions)
        
        #######################
        # Term 3
        #######################

        #self.prodSampledActsMSAAll = [self.sampledActs_spl[act]*self.MSA_out[act] for act in xrange(0, self.t.env_.numberOfActions)]

        # Current M values for sampled s and sampled a only. [Term 3 of loss function]
        self.MSA_sampled_act = tf.gather_nd(indices=self.Mextract, params=self.MSA_out)
        
        #self.MSA_sampled_act = tf.reduce_sum(self.prodSampledActsMSAAll, axis=0)
        
        #######################
        # Term 2
        #######################
        
#         self.sampNextAct = tf.placeholder(shape=[None, self.t.env_.numberOfActions], dtype=tf.float32)

#         self.sampNextAct_spl = tf.split(self.sampNextAct, self.t.env_.numberOfActions, axis=1)

        # Old M_s' values placeholder
        self.oldMSDV = tf.placeholder(shape=[self.t.env_.numberOfActions, None, self.featsSAS], dtype=tf.float32, name="oldMSDV")
        
        # Probabilities of Actions placeholder
        self.actProbSD = tf.placeholder(shape=[None, self.t.env_.numberOfActions], dtype=tf.float32, name="actProb")
        
        # Probabilities of Actions splitted for each action
        self.actProbSD_splitted = tf.split(self.actProbSD, self.t.env_.numberOfActions, axis=1)
        
        # Prod of Old M_s' and Probabilities of Actions
        self.prodActoldMSDV = [self.oldMSDV[act]*self.actProbSD_splitted[act] for act in xrange(0, self.t.env_.numberOfActions)]
        
        # Sum of products as in Term 2
        self.sumOfProds = tf.reduce_sum(self.prodActoldMSDV, axis=0)

        
        ##############################
        # Loss for learning M Values for Events
        ##############################

        self.Term1EventsPlaceholder = tf.placeholder(shape=[None, self.featsSAS_Events], dtype=tf.float32)
        
        self.OneMinusTerm1EventsPlaceholder = 1.0 - self.Term1EventsPlaceholder
        
        self.MSA_Events_sampled_act = tf.gather_nd(indices=self.Mextract, params=self.MSA_Events)
        
        self.oldMSDEventsV = tf.placeholder(shape=[self.t.env_.numberOfActions, None, self.featsSAS_Events], dtype=tf.float32, name="oldMSDV")
        
        self.prodActoldMSDEventsV = [self.oldMSDEventsV[act]*self.actProbSD_splitted[act] for act in xrange(0, self.t.env_.numberOfActions)]
        
        self.sumOfProdsEvents = tf.reduce_sum(self.prodActoldMSDEventsV, axis=0)
        
        self.sumOfProdsEventsMul1MIE = self.OneMinusTerm1EventsPlaceholder * self.sumOfProdsEvents

        ##############################
        # Total Loss for learning M values
        ##############################

        self.loss2_MDP = tf.reduce_mean(tf.square(tf.norm(self.Term1Placeholder + self.sumOfProds - self.MSA_sampled_act, ord='euclidean', axis=1)))

        self.loss2_Events = tf.reduce_mean(tf.square(tf.norm(self.Term1EventsPlaceholder + self.sumOfProdsEventsMul1MIE - self.MSA_Events_sampled_act, ord='euclidean', axis=1)))

        self.loss2 = self.loss2_MDP + self.loss2_Events

        # Learning Rate
        self.learning_rate2 = self.c.MlearningRate
        
        # Defining Optimizer
        self.optimizer2 = tf.train.AdamOptimizer(self.learning_rate2)
        
        # Update Gradients
        self.learning_step2 = (self.optimizer2.minimize(self.loss2))

        ##############################
        # Total Loss for learning Policy
        ##############################
        
#         self.MSAPol = tf.placeholder(shape=[self.t.env_.numberOfActions, None, self.featsSAS], dtype=tf.float32)
#         self.WeightPol = tf.placeholder(shape=[self.featsSAS], dtype=tf.float32)
        
#         self.MSAWPol = tf.transpose(tf.reduce_sum(tf.multiply(self.MSAPol, self.WeightPol), axis=2))
        self.MSAWPol = tf.placeholder(shape=[None, self.t.env_.numberOfActions], dtype=tf.float32)
        
        self.policyVal = self.policy.l_prob
        self.piMMSAW = tf.multiply(self.policyVal, self.MSAWPol)
        self.sumA_piMMSAW = tf.reduce_sum(self.piMMSAW, axis=1)
        
        ##############################
        # Total Loss for learning Policy for events
        ##############################
        
        self.startStateExtractor = tf.placeholder(shape=[None, 1], dtype=tf.int32)
        #self.startactions = tf.placeholder(shape=[self.t.batch, self.t.env_.numberOfActions], dtype=tf.float32)
        
        self.MEventsCurrent_sampled_act = tf.gather_nd(indices=self.Mextract, params=self.MSA_Events)
        self.MEventsCurrent_sampled_act_st = tf.gather_nd(indices=self.startStateExtractor, params=self.MEventsCurrent_sampled_act)

        self.logOfpolicyVal = tf.log(self.policyVal)
        self.logOfpolicyValSampledActions = tf.reduce_sum(tf.multiply(self.logOfpolicyVal, self.sampledActs), axis=1)
#         self.logOfpolicyVal_StStates_StActions = tf.gather_nd(indices=self.startStateExtractor, params=self.logOfpolicyValSampledActions)
#         self.logOfpolicyVal_StStates_StActions = tf.reshape(self.logOfpolicyVal_StStates_StActions, shape=[self.t.batch, 1])
        self.logOfpolicyVal_StStates_StActions = [self.logOfpolicyValSampledActions[bh] for bh in xrange(0, self.t.batch*self.t.deli, self.t.deli)]
        self.logOfpolicyVal_StStates_StActions = tf.reshape(self.logOfpolicyVal_StStates_StActions, shape=[self.t.batch, 1])

        self.Placeholder_MESSSA = tf.placeholder(shape=[self.t.batch, self.featsSAS_Events], dtype=tf.float32)

        self.MEventsCurrentlogPolicy = tf.multiply(self.logOfpolicyVal_StStates_StActions, self.Placeholder_MESSSA)
        self.ExpecMEventsCurrentlogPolicy = tf.reduce_mean(self.MEventsCurrentlogPolicy, axis=0)
        
        self.constraint_rew_placeholder = tf.placeholder(shape=[self.t.numberOfConstraints], dtype=tf.float32)
        self.constraint_prod_placeholder = tf.placeholder(shape=[self.t.numberOfConstraints], dtype=tf.float32)
        
        self.LogPolicyRightPartGradientSliced = [self.logOfpolicyValSampledActions[bh+1:bh+self.t.deli] for bh in xrange(0, self.t.batch*self.t.deli, self.t.deli)]
        self.LogPolicyRightPartGradientSliced = tf.reshape(self.LogPolicyRightPartGradientSliced, shape=[self.t.batch, (self.t.deli)-1, 1])
        
        self.indicatorProducts = tf.placeholder(shape=[self.t.batch, self.t.deli-1, self.featsSAS_Events], dtype=tf.float32)
        self.MEventsRightPartGradientSliced = tf.placeholder(shape=[self.t.batch, self.t.deli-1, self.featsSAS_Events], dtype=tf.float32)
        self.RightPartGradientTerm2Product1 = self.indicatorProducts * self.MEventsRightPartGradientSliced
        self.RightPartGradientTerm2FinalProduct = [self.LogPolicyRightPartGradientSliced[bh]*self.RightPartGradientTerm2Product1[bh] for bh in xrange(self.t.batch)]
        self.RightPartGradientTerm2SumT = tf.reduce_sum(self.RightPartGradientTerm2FinalProduct, axis=1)
        self.ExpectedRightPartGradientTerm2SumT = tf.reduce_mean(self.RightPartGradientTerm2SumT, axis=0)
        
        self.RightPartEventGradient = tf.reduce_sum(self.RightPartEventGradient_allEv)
        
        self.surr_Events = self.leftPartEventGradient + self.RightPartEventGradient
        
        self.surr_mdp = tf.reduce_mean(self.sumA_piMMSAW)
        
        self.loss3 = -1*(self.surr_mdp)
        
        # Learning Rate
        self.learning_rate3 = self.c.PolMDPlearningRate
        
        # Defining Optimizer
        self.optimizer3 = tf.train.AdamOptimizer(self.learning_rate3)
        
        # Update Gradients
        self.learning_step3 = (self.optimizer3.minimize(self.loss3))


        self.learning_rate4 = self.c.PolEVlearningRate
        
        self.loss4 = -1*(self.surr_Events)
        
        self.optimizer4 = tf.train.AdamOptimizer(self.learning_rate4)

        self.learning_step4 = (self.optimizer4.minimize(self.loss4))

        return self



# In[ ]:


os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
conf = tf.ConfigProto()
conf.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
conf.gpu_options.allow_growth = True #allocate dynamically

class EDECMDP():

    def __init__(self, config):
        self.config = config

        self.numAgents = config.agents
        
    def _instance_method_alias_run_iteration(self, obj, session):
        return obj.runIteration(session)
    
    def _instance_method_alias_comp_prod(self, obj, inp1, inp2):
        return obj.computeProdProbability(inp1, inp2)

    def merge_two_dicts(self, x, y):
        z = x.copy()   # start with x's keys and values
        z.update(y)    # modifies z with y's keys and values & returns None
        return z

    def evalMVAlues(self, evaldict):

        # N * 3 action probabilities for each action of each sample s'
        actProb = evaldict["actProbSA"]
        
        # 3 * N * |F|, M_S' values
        MSVals = evaldict["MSV"]
        
        # weight vector |F|*1
        weights = evaldict["weights"]
        
        deli = evaldict["delimiter"]
        
        MSAW = np.transpose(np.sum(np.multiply(weights, MSVals), axis=2))

        MSAWActProb = np.multiply(actProb, MSAW)
        MSAWActProb = np.array(MSAWActProb)
        
        valuesAllSamples = np.sum(MSAWActProb, axis=1)
        
        valuesAllSamples = np.array(valuesAllSamples).reshape((self.config.batch, deli))
        
        valuesStartState = np.array([valuesAllSamples[vals][0] for vals in xrange(self.config.batch)])
        
        return np.mean(valuesStartState)
    
    def getAvgRewardFromEvents(self, test_maps):
        
        batch_size = self.config.batch
        
        batch_rewards = []
        for batch in xrange(0, batch_size):

            batch_rew  = 0.0

            for lindex in xrange(0, len(self.config.sharedSites)):

                loc = self.config.sharedSites[lindex]
                T = self.config.T
                actionTime = self.config.actionTimes[0]

                owners = []
                for line in self.config.lines:
                    if line.hasLocation(loc):
                        owners.extend(line.getOwners())

                for start in xrange(T, 0, -self.config.repeatKTimeSteps):
                    if (start - actionTime < 0):
                        continue

#                     print "Generating for event with location ", loc, " and Time range ", start, " - ", start - self.config.repeatKTimeSteps

                    for owner in owners:
                        map_vals = test_maps[owner]
                        assert map_vals["agent"] == owner
                        if map_vals[batch][loc][start] == True:
#                             print "Happened: event with location ", loc, " and Time range ", start, " - ", start - self.config.repeatKTimeSteps, " by: ", owner
#                             print
                            batch_rew += self.config.creward[lindex]
                            break
 
            batch_rewards.append(batch_rew)
        return batch_rewards
        
    def initializeRL(self, filename):
        
        f = open(self.config.workDir+'Logs/'+filename+'.csv', 'w', 0)
        f.write('Iteration,Time, ,')
        for ag in range(self.numAgents):
            f.write('Agent_'+str(ag)+'_Loss1_Initial,Agent_'+str(ag)+'_Loss1_Final, ,Agent_'+str(ag)+'_Loss2_Initial, Agent_'+str(ag)+'_Loss2_Final, ,Agent_'+str(ag)+'_Loss3_Initial,Agent_'+str(ag)+'_Loss3_Final, ,Agent_'+str(ag)+'_Iter_MDP_Policy, Agent_'+str(ag)+'_Iter_MValues, ,')
        f.write('Total_Iter_MDP_Policy, Total_Iter_MValues, M_Percent_of_Policy, ,Total_Iter_Event, Total_Iter_System, ,Total_MDP_KIter_Policy, Total_MValues_KIter_Policy, Total_Event_KIter_Policy, Total_System_KIter_Policy\n')

        # Holds train class instances for each agent
        train_instances = []

        # Holds placeholder class instances for each agent
        placeholderInstances = []

        # Holds Replica NN instances for each agent
        replicaNNInstances = []

        # Holds Replica NN input placeholder instances for each agent
        replicaNNPlInstances = []

        # Initialize train instances
        for ag in xrange(self.numAgents):
            train_instances.append(Train(agent=ag, config=self.config))

        # Initialize primary graphs
        g1 = tf.Graph()
        with g1.as_default():
            for ag in xrange(self.numAgents):
                p = PlaceholderClass(agent=ag, c=self.config, t=train_instances[ag])
                p = p.computationalGraphs()
                train_instances[ag].policyNN = p.policy
                train_instances[ag].NNInstance = p.NN
                assert train_instances[ag].policyNN != None
                placeholderInstances.append(p)
            sess_1 = tf.Session(graph=g1, config=conf)
            
        writer = tf.summary.FileWriter(self.config.workDir+'tflogs/'+filename, sess_1.graph)
        Iloss_writer = tf.summary.FileWriter(self.config.workDir+'tflogs/'+filename+"_InitialLoss")
        Floss_writer = tf.summary.FileWriter(self.config.workDir+'tflogs/'+filename+"_FinalLoss")

        g2 = tf.Graph()
        with g2.as_default():
            for ag in xrange(self.numAgents):
                t = train_instances[ag]
                inputsR = tf.placeholder(shape=[None, t.env_.validLengthOfState], dtype=tf.float32)
                replicaNNPlInstances.append(inputsR)
                if self.config.ReplayerNorm:
                    print "Defining Replica NN with LN"
                    instanceR = ReplicaNNWLN(featureSAS=t.featsSAS, featsSAS_Events=t.featsSAS_Events, actionCount=t.env_.numberOfActions, input_var=inputsR, agent=ag, config=self.config)
                else:
                    print "Defining Replica NN without LN"
                    instanceR = ReplicaNN(featureSAS=t.featsSAS, featsSAS_Events=t.featsSAS_Events, actionCount=t.env_.numberOfActions, input_var=inputsR, agent=ag, config=self.config)
                replicaNNInstances.append(instanceR)
            sess_2 = tf.Session(graph=g2, config=conf)

        tf.reset_default_graph()

        # Used to save and restore policy from NNs
        policySavers = []

        # Used to save and restore M values from NNs
        MValSavers = []

        # Used to save and restore M Replica values from NNs
        MRelpicaValSavers = []

        pool = Pool(processes=(self.numAgents+2))
        
        stopwatch = Stopwatch()

        with g1.as_default():
            sess_1.run(tf.global_variables_initializer())
            all_vars = []
            for v in tf.trainable_variables():
                if "State_Encoder" in v.name or "State_Decoder" in v.name or "Phi_Encoder" in v.name or "MSA" in v.name or "Reward_Weights" in v.name:
                    all_vars.append(v)
#                     print v
            OverallSaver = tf.train.Saver(all_vars)
            
            if self.config.runWithSavedModel==True:
                print "Saving from ", self.config.loadPolicyFileFull
                OverallSaver.restore(sess_1, str(self.config.loadPolicyFileFull))
                print("Overall Model restored.")

            orig_loss1 = [placeholderInstances[j].loss1 for j in xrange(self.numAgents)]
            learn_loss1 = [placeholderInstances[j].learning_step1 for j in xrange(self.numAgents)]
            new_loss1 = [placeholderInstances[j].loss1 for j in xrange(self.numAgents)]

            orig_loss2 = [placeholderInstances[j].loss2 for j in xrange(self.numAgents)]
            learn_loss2 = [placeholderInstances[j].learning_step2 for j in xrange(self.numAgents)]
            new_loss2 = [placeholderInstances[j].loss2 for j in xrange(self.numAgents)]
            
            orig_loss3 = [placeholderInstances[j].loss3 for j in xrange(self.numAgents)]
            learn_loss3 = [placeholderInstances[j].learning_step3 for j in xrange(self.numAgents)]
            new_loss3 = [placeholderInstances[j].loss3 for j in xrange(self.numAgents)]
                        
            orig_loss4 = [placeholderInstances[j].loss4 for j in xrange(self.numAgents)]
            learn_loss4 = [placeholderInstances[j].learning_step4 for j in xrange(self.numAgents)]
            new_loss4 = [placeholderInstances[j].loss4 for j in xrange(self.numAgents)]

            # Used to compute term 1 of Loss 2
            PhiSAS_sampled_act_AllAgents = [placeholderInstances[j].PhiSAS_sampled_act for j in xrange(self.numAgents)]
            
            for ag in range(self.numAgents):

                # Restoring Policy
#                 var = [v for v in tf.trainable_variables() if "Policy_Agent_"+str(ag) in v.name]
#                 saverAndRestorePolicy = tf.train.Saver(var)
#                 saverAndRestorePolicy.restore(sess_1, self.config.loadPolicyFile)
#                 policySavers.append(saverAndRestorePolicy)
#                 print("Policy Model for Agent "+str(ag)+" restored.")

                # Saver for M network
                Mvar = [v for v in tf.trainable_variables() if "MSA_Agent_"+str(ag) in v.name]
                MSA_saverAndRestore = tf.train.Saver(Mvar)
                MSA_saverAndRestore.save(sess_1, self.config.saveMValuesDir+str(filename)+"_MValues_Agent_"+str(ag)+".ckpt")
                MValSavers.append(MSA_saverAndRestore)
                print("Original M Values for Agent "+str(ag)+" saved.")

            # Restore M values in replica network
            with g2.as_default():
                sess_2.run(tf.global_variables_initializer())

                for ag in range(self.numAgents):
                    MRvar = [v for v in tf.trainable_variables() if "MSA_Agent_"+str(ag) in v.name]
                    MReplica_saverAndRestore = tf.train.Saver(MRvar)
                    MReplica_saverAndRestore.restore(sess_2, self.config.saveMValuesDir+str(filename)+"_MValues_Agent_"+str(ag)+".ckpt")
                    MRelpicaValSavers.append(MReplica_saverAndRestore)
                    print("Replica M Values for Agent "+str(ag)+" restored.")

            IterMDPTotal = []
            IterMValsTotal = []
            IterEventTotal = []
            IterSystemTotal = []
            
            for it in range(1, self.config.numIterations+1):
#                 print "\n================Iteration "+str(it)+": \n"
                
                ##########################
                # Annealing
                ##########################
                if self.config.annealRandomActProb:
                    self.config.randomActionProb = 1.0/(2.0*math.sqrt(it))
                    print "Random Action Probability: ", self.config.randomActionProb
                
                ##########################
                # Sampling
                ##########################
                event_maps = []
  
                res = pool.amap(self._instance_method_alias_run_iteration, train_instances, [sess_1]*self.numAgents)
                train_instances = res.get()

                for j in xrange(0, self.numAgents):
                    event_maps.append(train_instances[j].eventMap)
                    
                feeding_dicts = []
                for ag in xrange(self.numAgents):
                    t = train_instances[ag]
                    feeding_d = t.giveMeEverything(placeholderInstances[ag])                    
                    feeding_dicts.append(feeding_d)

                assert self.numAgents >= 2
                merged_dicts_feeding = self.merge_two_dicts(feeding_dicts[0], feeding_dicts[1])
                for ag in xrange(2, self.numAgents):
                    merged_dicts_feeding = self.merge_two_dicts(merged_dicts_feeding, feeding_dicts[ag])

                ##########################
                # Learn Loss 1
                ##########################
                initLoss1 = sess_1.run(orig_loss1, feed_dict=merged_dicts_feeding)
                sess_1.run(learn_loss1, feed_dict=merged_dicts_feeding)
                finLoss1 = sess_1.run(orig_loss1, feed_dict=merged_dicts_feeding)                
#                 print "==Loss_1: ",
#                 print initLoss1, 
#                 print " ==> ",
#                 print finLoss1

                PhiSAS_sampled_act_AllAgents_Vals = sess_1.run(PhiSAS_sampled_act_AllAgents, feed_dict=merged_dicts_feeding)

                with g2.as_default():
                    for ag in xrange(self.numAgents):
                        t = train_instances[ag]
                    
                        replicaFeed = dict()
                        instanceR = replicaNNInstances[ag]
                        replicaFeed[instanceR.input_var] = t.finalPATHS["nextobservations"]
                        oldMSDVals, oldMSDEventVals = sess_2.run((instanceR.replica_msa_values, instanceR.replica_msa_events_values), feed_dict=replicaFeed)
                        oldMSDVals = np.array(oldMSDVals)
                        oldMSDEventVals = np.array(oldMSDEventVals)

                        p = placeholderInstances[ag]
                        merged_dicts_feeding[p.oldMSDV] = oldMSDVals
                        merged_dicts_feeding[p.Term1Placeholder] = PhiSAS_sampled_act_AllAgents_Vals[ag]
                        merged_dicts_feeding[p.MInput_var] = t.finalPATHS["observations"]
                        merged_dicts_feeding[p.oldMSDEventsV] = oldMSDEventVals

                ##########################
                # Learn Loss 2
                ##########################
                initLoss2 = sess_1.run(orig_loss2, feed_dict=merged_dicts_feeding)
                sess_1.run(learn_loss2, feed_dict=merged_dicts_feeding)
                finLoss2 = sess_1.run(orig_loss2, feed_dict=merged_dicts_feeding)
#                 print "==Loss_2: ",
#                 print initLoss2, 
#                 print " ==> ",
#                 print finLoss2

                MSAEvPolSSSA_AllAgents = []
                for ag in xrange(self.numAgents):
                    p = placeholderInstances[ag]
                    t = train_instances[ag]

                    #merged_dicts_feeding[p.policy.input_var] = encoded_S_AllAgents_Vals[ag]
                    #merged_dicts_feeding[p.policy.train_mode] = True
                    MSAPol, MSAEvPolSSSA, WeightPol, MSAEvPolSamA = np.array(sess_1.run((p.MSA_out, p.MEventsCurrent_sampled_act_st, p.r_weight, p.MEventsCurrent_sampled_act), feed_dict=merged_dicts_feeding))
                    MSAWPolVal = np.transpose(np.sum(np.multiply(MSAPol, WeightPol), axis=2))
                    
                    if self.config.normalizeMSAW:
                        MSAWPolVal = (MSAWPolVal-np.mean(MSAWPolVal))/(np.std(MSAWPolVal)+1e-8)

                    merged_dicts_feeding[p.MSAWPol] = MSAWPolVal
                    merged_dicts_feeding[p.Placeholder_MESSSA] = MSAEvPolSSSA
                    MSAEvPolSSSA_AllAgents.append(MSAEvPolSSSA)
                    
                    MEventsRightPartGradientSlicedVal = [MSAEvPolSamA[bh+1:bh+t.deli,:] for bh in xrange(0, t.batch*t.deli, t.deli)]
                    MEventsRightPartGradientSlicedVal = np.array(MEventsRightPartGradientSlicedVal)
                    merged_dicts_feeding[p.MEventsRightPartGradientSliced] = MEventsRightPartGradientSlicedVal

                prods = pool.amap(self._instance_method_alias_comp_prod, train_instances, [event_maps]*self.numAgents ,[MSAEvPolSSSA_AllAgents]*self.numAgents)
                exp_prods = prods.get()
                
                for ag in xrange(self.numAgents):
                    t = train_instances[ag]
                    p = placeholderInstances[ag]
                    
                    merged_dicts_feeding[p.constraint_prod_placeholder] = np.array(exp_prods[ag])
                    cons_rew_ag = np.array(t.constraints_rewards)
                    
                    if self.config.normalizeCK:
                        cons_rew_ag = (cons_rew_ag - np.mean(cons_rew_ag)) / (np.std(cons_rew_ag) + 1e-8)

                    merged_dicts_feeding[p.constraint_rew_placeholder] = np.array(cons_rew_ag)   
                            
                ##########################
                # Learn Loss 3
                ##########################
                initLoss3 = sess_1.run(orig_loss3, feed_dict=merged_dicts_feeding)
                sess_1.run(learn_loss3, feed_dict=merged_dicts_feeding)
                finLoss3 = sess_1.run(orig_loss3, feed_dict=merged_dicts_feeding)

#                 print "==Loss_3: ",
#                 print initLoss3, 
#                 print " ==> ",
#                 print finLoss3

                initLoss4 = sess_1.run(orig_loss4, feed_dict=merged_dicts_feeding)
                sess_1.run(learn_loss4, feed_dict=merged_dicts_feeding)
                finLoss4 = sess_1.run(orig_loss4, feed_dict=merged_dicts_feeding)

#                 print "==Loss_4: ",
#                 print initLoss4, 
#                 print " ==> ",
#                 print finLoss4

                elapTime = stopwatch.elapsedTime()
#                 print '\tElapsed Time: ', elapTime
                avg_time_per_iter = elapTime/it
#                 print '\tAverage Time per Iteration: ', avg_time_per_iter

                ##########################
                # Evaluate learned policy
                ##########################
                MDPValsEval = []
                MValsEval = []
                TestMapsAllAgents = []

                for ag in range(self.numAgents):
                    t = train_instances[ag]
                    assert t.finalPATHS != None
                    mdp_policy_eval = t.finalPATHS["avgMDPValue"]
                    MDPValsEval.append(mdp_policy_eval)

                    evaluator = dict()
                    t = train_instances[ag]
                    p = placeholderInstances[ag]

                    evaluator["actProbSA"] = t.finalPATHS["nextActionProbsS"]
                    evaluator["MSV"], evaluator["weights"] = np.array(sess_1.run((p.MSA_out, p.r_weight), feed_dict={p.MInput_var: t.finalPATHS["observations"]}))
                    evaluator["delimiter"] = int(float(t.env_.T) / float(t.env_.actionTime))
                    Mvalueeval = self.evalMVAlues(evaluator)

                    MValsEval.append(Mvalueeval)
                    TestMapsAllAgents.append(t.testMap)

                totalMDP = np.sum(np.array(MDPValsEval))
                IterMDPTotal.append(totalMDP)
                KIterMDPTotal = np.mean(IterMDPTotal[self.config.lastKIter:])
                totalM = np.sum(np.array(MValsEval))
                IterMValsTotal.append(totalM)
                KIterMValsTotal = np.mean(IterMValsTotal[self.config.lastKIter:])
                
                eventRewards = self.getAvgRewardFromEvents(TestMapsAllAgents)
                eventRewards = np.array(eventRewards)
                
                totalEvent = np.mean(eventRewards)
                IterEventTotal.append(totalEvent)
                KIterEventTotal = np.mean(IterEventTotal[self.config.lastKIter:])
                
                totalSystem = totalMDP+totalEvent
                IterSystemTotal.append(totalSystem)
                KIterSystemTotal = np.mean(IterSystemTotal[self.config.lastKIter:])

#                 print "Agent 0 Iter MDP ", train_instances[0].finalPATHS["avgMDPValue"]
#                 print "Agent 1 Iter MDP ", train_instances[1].finalPATHS["avgMDPValue"]
#                 print "Total MDP ", totalMDP
#                 print "Total M Values ", totalM
#                 print "Total Event Value", totalEvent
#                 print "Total System Value", totalSystem
#                 print "Percent ", (100*(totalM/totalMDP))
#                 print "last K Iter MDP ", KIterMDPTotal
#                 print "last K Iter MValues ", KIterMValsTotal
#                 print "last K Iter Event ", KIterEventTotal
#                 print "last K Iter System ", KIterSystemTotal
#                 print

                ##########################
                # Restrictive Updates & Logging
                ##########################
                if it%self.config.MValuesPeriodicUpdateEveryXIter==0:
                    for ag in range(self.numAgents):
                        MValSavers[ag].save(sess_1, self.config.saveMValuesDir+str(filename)+"_MValues_Agent_"+str(ag)+".ckpt", write_meta_graph=False)
                        print("Original M Values for Agent "+str(ag)+" saved.")
                        with g2.as_default():
                            MRelpicaValSavers[ag].restore(sess_2, self.config.saveMValuesDir+str(filename)+"_MValues_Agent_"+str(ag)+".ckpt")
                            print("Replica M Values for Agent "+str(ag)+" restored.")
                    print
                    
                if it%self.config.savingEveryXIter==0:
                    OverallSaver.save(sess_1, self.config.saveModelsDir+str(filename)+".ckpt", write_meta_graph=False)
                    print("Overall Model saved.")
            
                if it >= self.config.loggingThreshold:

                    summary = tf.Summary()               
                    summary2 = tf.Summary()
                    summary3 = tf.Summary()
                    summary.value.add(tag='summaries/KIterMDPTotal', simple_value = KIterMDPTotal)
                    summary.value.add(tag='summaries/KIterMValsTotal', simple_value = KIterMValsTotal)
                    summary.value.add(tag='summaries/KIterEventTotal', simple_value = KIterEventTotal)
                    summary.value.add(tag='summaries/KIterSystemTotal', simple_value = KIterSystemTotal)
                    summary.value.add(tag='summaries/avg_time_per_iteration', simple_value = avg_time_per_iter)
                    for ag in range(self.numAgents):

                        summary2.value.add(tag='losses/Loss_1'+str(ag), simple_value = initLoss1[ag])
                        summary2.value.add(tag='losses/Loss_2'+str(ag), simple_value = initLoss2[ag])
                        summary2.value.add(tag='losses/Loss_3'+str(ag), simple_value = initLoss3[ag])
                        summary2.value.add(tag='losses/Loss_4'+str(ag), simple_value = initLoss4[ag])

                        summary3.value.add(tag='losses/Loss_1'+str(ag), simple_value = finLoss1[ag])
                        summary3.value.add(tag='losses/Loss_2'+str(ag), simple_value = finLoss2[ag])
                        summary3.value.add(tag='losses/Loss_3'+str(ag), simple_value = finLoss3[ag])
                        summary3.value.add(tag='losses/Loss_4'+str(ag), simple_value = finLoss4[ag])

                    writer.add_summary(summary, it)
                    writer.flush()
                    Iloss_writer.add_summary(summary2, it)
                    Iloss_writer.flush()
                    Floss_writer.add_summary(summary3, it)
                    Floss_writer.flush()

                    f.write(str(it)+','+str(elapTime)+', ,')
                    for ag in range(self.numAgents):
                        f.write(str(initLoss1[ag])+","+str(finLoss1[ag])+", ,"+str(initLoss2[ag])+","+str(finLoss2[ag])+", ,"+str(initLoss3[ag])+","+str(finLoss3[ag])+", ,"+str(MDPValsEval[ag])+","+str(MValsEval[ag])+", ,")
                    f.write(str(totalMDP)+","+str(totalM)+","+str(100*(float(totalM)/float(totalMDP)))+", ,"+str(totalEvent)+","+str(totalSystem)+", ,"+str(KIterMDPTotal)+","+str(KIterMValsTotal)+","+str(KIterEventTotal)+","+str(KIterSystemTotal)+'\n')
            f.close()
            pool.close()


# In[ ]:


# c = Config(1001)
# c.normalizeCK = True
# c.normalizeMSAW = True
# e = EDECMDP(c)
# e.initializeRL(filename="Testing1001.1Scratch")
# from tensorflow.python.tools import inspect_checkpoint as chkp
# chkp.print_tensors_in_checkpoint_file(file_name="../../RL-PYNB/modelsO/model_200_T9_32.ckpt", tensor_name="encoder", all_tensors=False, all_tensor_names=True)


# In[ ]:


# from tensorflow.python.tools import inspect_checkpoint as chkp
# chkp.print_tensors_in_checkpoint_file(file_name=c.loadPolicyFileFull, tensor_name="encoder", all_tensors=False, all_tensor_names=True)

