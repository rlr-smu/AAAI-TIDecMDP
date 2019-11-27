import tensorflow as tf
import numpy as np
from Env import Env
from Config import Config
from NNPolicy import ANNPol
from ValueFunction import VFLearn
from NNPolicyWLN import ANNPolWLN
from ValueFunctionWLN import VFLearnWLN
from Train import Train
import json

class PlaceholderClass:
    
    """
    t -> Train Instance
    c -> Config Instance
    agent -> Which agent?!
    """
    def __init__(self, t, c, agent):
        
        # -----------------------------------------------------------------------------------------------------
        # Baseline NN Computational Graph Placeholders
        # -----------------------------------------------------------------------------------------------------
        
        # Train Instance
        self.t = t
        
        # Config Instance
        self.c = c
        
        # Which agent?
        self.agent = agent
        
        # Input placeholder for baseline
        self.input_var_baseline  = tf.placeholder(shape=[None, t.env_.validLengthOfState], dtype=tf.float32, name="input_var_baseline_"+str(agent))

        # Baseline Train Mode? 
        self.train_mode_baseline = tf.placeholder(tf.bool, name="train_mode_baseline_"+str(agent))
        
        # Baseline Object
        if self.c.VFLayerNorm:
            print "Defining VF Network With Layer Norm"
            self.baseline = VFLearnWLN(t.env_.validLengthOfState, self.input_var_baseline, self.train_mode_baseline, c, self.agent)
        else:
            print "Defining VF Network Without Layer Norm"
            self.baseline = VFLearn(t.env_.validLengthOfState, self.input_var_baseline, self.train_mode_baseline, c, self.agent)
        
        # Target VF
        self.targets = tf.placeholder(shape=[t.batch * int(float(t.env_.T) / float(t.env_.actionTime)), 1], dtype=tf.float32, name="target_baseline_"+str(agent))

        # -----------------------------------------------------------------------------------------------------
        # Policy NN Computational Graph Placeholders
        # -----------------------------------------------------------------------------------------------------

        # Placeholder for storing array index slicing
        self.arr_placeholder = tf.placeholder(shape=[t.numberOfConstraints+1, 1], dtype=tf.int32, name="arr_placeholder_"+str(agent))
        
        # Whether a constraint should be included.
        self.constraint_included = tf.placeholder(shape=[t.numberOfConstraints, 1], dtype=tf.float32, name="constraint_included_"+str(agent))

        # Input placeholder for policy NN
        self.input_var_policy  = tf.placeholder(shape=[None, t.env_.validLengthOfState], dtype=tf.float32, name="input_var_policy_"+str(agent))
        
        # Action values placeholder for policy NN
        self.action_var_policy = tf.placeholder(shape=[None, t.env_.numberOfActions], dtype=tf.float32, name="action_var_policy"+str(agent))
        
        # Whether train mode?
        self.train_mode_policy = tf.placeholder(tf.bool, name="train_mode_policy_"+str(agent))

        # Policy Object
        if self.c.PolLayerNorm:
            print "Defining Policy Network With Layer Norm"
            self.policy = ANNPolWLN(t.env_.validLengthOfState, t.env_.numberOfActions, self.input_var_policy, self.train_mode_policy, self.agent, c)                   
        else:
            print "Defining Policy Network Without Layer Norm"
            self.policy = ANNPol(t.env_.validLengthOfState, t.env_.numberOfActions, self.input_var_policy, self.train_mode_policy, self.agent, c)       

        self.returns_var = tf.placeholder(shape=[None, t.batch * (int(float(t.env_.T) / float(t.env_.actionTime)))], dtype=tf.float32, name="returns_var_policy_"+str(agent))

        self.rew_scalars = tf.placeholder(shape=[t.numberOfConstraints, 1], dtype=tf.float32, name="rew_scalars_"+str(agent))
        self.prod_scalars = tf.placeholder(shape=[t.numberOfConstraints, 1], dtype=tf.float32, name="prod_scalars_"+str(agent))
        
        # -----------------------------------------------------------------------------------------------------
        # Baseline NN Computational Graph
        # -----------------------------------------------------------------------------------------------------

        self.predictions = None
        self.loss_baseline = None
        self.learning_rate_baseline = None
        self.baseline_optimizer = None
        self.learning_step_baseline = None
        
        self.all_outputs = None
        self.surr_mdp = None
        self.splitted_log_likelihood = None
        self.val_for_constraints = None
        self.surr_events = None
        self.loss = None
        self.learning_rate = None
        self.optimizer = None
        self.learning_step = None
        
    def computationalGraphs(self):
        # -----------------------------------------------------------------------------------------------------
        # Baseline NN Computational Graph
        # -----------------------------------------------------------------------------------------------------

        # Predictions of current VF
        self.predictions = self.baseline.l_vf
        
        # Loss Function for the Value Function
        self.loss_baseline = tf.losses.mean_squared_error(predictions=self.predictions, labels=self.targets)

        # Learning Rate
        self.learning_rate_baseline = self.t.config.BaslinelearningRate
        
        # Initialize Optimizer
        self.baseline_optimizer = tf.train.AdamOptimizer(self.learning_rate_baseline)

        # Update Gradients
        self.learning_step_baseline = (self.baseline_optimizer.minimize(self.loss_baseline))
        
        
        # -----------------------------------------------------------------------------------------------------
        # Policy NN Computational Graph
        # -----------------------------------------------------------------------------------------------------

        self.all_outputs = self.policy.log_likeli_sym(self.action_var_policy)
        
        # Loss function for MDP
        self.surr_mdp = tf.reduce_mean(self.all_outputs[0:self.arr_placeholder[0,0]] * self.returns_var)
        
        self.splitted_log_likelihood = [tf.multiply(tf.reduce_mean(self.all_outputs[self.arr_placeholder[i,0]:self.arr_placeholder[i+1,0]]), self.constraint_included[i,0]) for i in xrange(0,self.t.numberOfConstraints)]
        nan_values = tf.is_nan(self.splitted_log_likelihood)
        self.splitted_log_likelihood = tf.where(tf.is_nan(self.splitted_log_likelihood), tf.zeros_like(self.splitted_log_likelihood), self.splitted_log_likelihood)
        
        self.val_for_constraints = [self.rew_scalars[i,0] * self.splitted_log_likelihood[i] * self.prod_scalars[i,0] for i in xrange(0,self.t.numberOfConstraints)]
        
        # Loss function for Events
        self.surr_events = tf.reduce_sum(self.val_for_constraints)
        
        self.loss = -1*(self.surr_mdp + self.surr_events)
        self.learning_rate = self.t.config.PolicylearningRate
        
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        
        # Update Gradients
        self.learning_step = (self.optimizer.minimize(self.loss))
        
        return self
        
        


# In[ ]:


# c = Config(200)
# t = Train(0, c)

# pl = PlaceholderClass(t,c,0)
# pl = pl.computationalGraphs()
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print sess.run(pl.train_mode_baseline, feed_dict={pl.train_mode_baseline: True})

