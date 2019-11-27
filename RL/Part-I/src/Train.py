
# coding: utf-8

# In[1]:

import os
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from Env import Env
from Config import Config
from NNPolicy import ANNPol
from ValueFunction import VFLearn
import json


# In[2]:


class Train:

    def __init__(self, agent, config):
        
        # Configuration file object.
        self.config = config
        
        # Line of agent.
        self.line = self.config.agentWiseLines[agent]
        
        # All locations (shared/private) assigned to agent.
        self.locations = self.line.getLocations()
        
        # Create stilmulator for the agent.
        self.env_ = Env(agent, self.config)
        
        # Training class for the corresponding agent.
        self.agent = agent
        
        # Dict used to store data values for actual training of MDP
        # Stores obs, new_obs, rewards, etc.
        self.PATHS = None
        
        # Dict used to store data values for actual training of Events
        # Stores product scalar, constraint reward scalar, vectors of obs, acts for log_likeli_sym
        self.eventPATHS = None
        self.finalPATHS = None
        
        # Batch size to be used for training.
        self.batch = self.config.batch
        
        # First component used for computing event gradients, actual rewards c_k
        self.eventMap = dict()
        self.reset_times = []
        self.resetMappingFunction(updateResetTimes=True)
        self.constraints_rewards = self.constraintRewardsVector()
        self.numberOfConstraints = len(self.constraints_rewards)
        print self.constraints_rewards
        print self.reset_times
        
        # Third component for computing event gradients, Prod P(E_k^j)
        self.products = None
        
        self.policy = None
        self.baseline = None
        
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
        
    def resetMappingFunction(self, updateResetTimes = False):

        self.eventMap["agent"] = self.agent
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

                if (start - self.config.repeatKTimeSteps >= 0):
                    end = start - self.config.repeatKTimeSteps
                else:
                    end = 0

                while(action > end):
                    self.eventMap[loc][start][action] = dict()
                    self.eventMap[loc][start][action]["sample"] = []
                    self.eventMap[loc][start][action]["decision"] = -1

                    action -= self.config.actionTimes[self.agent]

            updateResetTimes = False
        #print json.dumps(self.eventMap, indent=6, sort_keys=True)
        
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
        
    def fillMappingFunction(self, PATHS):
        for ind in xrange(0, len(PATHS)):

            path = PATHS[ind]

            # print "=============Episode=========\n"

            observations = path["observations"]
            actions = path["actions"]
            nextObs = path["nextObs"]

            assert len(observations) == len(actions) == len(nextObs)
            length = len(observations)

            for j in xrange(0, length):
                obs = observations[j].tolist()
                next_obs = nextObs[j].tolist()
                act = actions[j]

                (location_obs, time_obs, inspection_obs, dold_obs, actualLocation_obs) = self.getDataForObs(obs)
                (location_nextobs, time_nextobs, inspection_nextobs, dold_nextobs, actualLocation_nextobs) = self.getDataForObs(next_obs)

                action_taken = self.env_.getAction(act)
                if action_taken == 0 and actualLocation_obs == actualLocation_nextobs and location_obs == location_nextobs and self.line.isShared(actualLocation_obs):

                    if (inspection_obs[location_obs] == 0 and dold_obs == 0):

                        if (inspection_nextobs[location_nextobs] == 1 and dold_nextobs == 0):

                            # Stores the number of times the event happens.
                            self.eventMap[actualLocation_obs][self.getTimeSlot(time_obs)]["val"] += 1
                            
                            # Stores the batches in which the events happens.
                            self.eventMap[actualLocation_obs][self.getTimeSlot(time_obs)][time_obs]["sample"].append(ind)
                            
                            # Stores the decision time at which the event happens.
                            self.eventMap[actualLocation_obs][self.getTimeSlot(time_obs)][time_obs]["decision"] = j

            
        # print json.dumps(self.eventMap, indent=4, sort_keys=True)
        
    def runIteration(self, sess):

        PATHS = []

        # store sampled states
        observations = []

        # store sampled actions i
        actions = []

        # Empirical return for each sampled state-action pair
        rewards = []

        # Keep track of the states sampled after the current state
        nextObs = []

        episodeEnd = False
        s = self.env_.reset()

        '''sample "mini-batch" trajectories'''
        while (episodeEnd != True):

            s = np.array(s).reshape((self.batch, self.env_.validLengthOfState))

            randomOrPolicy = np.random.choice(2, 1, p=[self.config.randomActionProb, 1-self.config.randomActionProb])
            if randomOrPolicy==0:
                a = [np.random.choice(self.env_.numberOfActions, 1) for i in xrange(self.batch)]
            elif randomOrPolicy==1:
                probs = self.policy.l_prob.eval({self.policy.input_var: s, self.policy.train_mode: False}, session=sess)
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

        # empirical return for each state-aciton or Q(s, a)
        returns = []

        # advantage function would be used with baseline A(s, a) = Return(s, a) - V(s)
        advantages = []
        rewards = rewards.tolist()
        for b in xrange(self.batch):
            ret = []
            adv = []
            return_so_far = 0
            for t in range(len(rewards[b]) - 1, -1, -1):
                return_so_far = (rewards[b][t] + return_so_far)
                ret.append(return_so_far)
            ret = ret[::-1]
            # Compute advantage for each sampled state-action
            for t in range(len(ret)):
                obs_baseline = np.array(observations[b][t]).reshape((1, self.env_.validLengthOfState))
                v_s = self.baseline.get_value(obs_baseline, sess)[0,0]
                adv.append(ret[t] - v_s) 
            returns.append(ret)
            advantages.append(adv)
        rewards = np.array(rewards)
        returns = np.array(returns)
        advantages = np.array(advantages)

        for bh in xrange(self.batch):
            # path is a record of a single complete T-step trajectory
            path = dict(
                observations=np.array(observations[bh]),
                nextObs=np.array(nextObs[bh]),
                actions=np.array(actions[bh]),
                rewards=np.array(rewards[bh]),
                returns=np.array(returns[bh]),
                advantages=np.array(advantages[bh])
            )

            # Append the entire trajectory to PATHS variable
            PATHS.append(path)
            
        self.PATHS = PATHS
        self.resetMappingFunction()
        self.fillMappingFunction(PATHS)
        
        assert len(PATHS) == self.batch
        
        # collect all states in an array with dimension N*S
        obs = np.concatenate([p["observations"] for p in PATHS])
        obs = np.array(obs).reshape((self.batch * int(float(self.env_.T) / float(self.env_.actionTime)), self.env_.validLengthOfState))
                
        # collect all actions, dimension N*A
        act = np.concatenate([p["actions"] for p in PATHS])      
        
        # collect all next states in an array with dimension N*S
        nextobs = np.concatenate([p["nextObs"] for p in PATHS])
        nextobs = np.array(nextobs).reshape((self.batch * int(float(self.env_.T) / float(self.env_.actionTime)), self.env_.validLengthOfState))

        # Collect all rewards
        rew = np.concatenate([p["rewards"] for p in PATHS])
        
        # all returns, single dimension vector (N,)
        ret = np.concatenate([p["returns"] for p in PATHS])   
        
        # all advantages, single dimension vector (N,)
        advn = np.concatenate([p["advantages"] for p in PATHS])    
        advn = np.array(advn).reshape((1, self.batch * int(float(self.env_.T) / float(self.env_.actionTime))))
        
        # Target function for the baseline
        targets = []
        targets = np.zeros(rew.shape)
        
        # Get value function for next obs to compute targets.
        vals = self.baseline.get_value(nextobs, sess)

        # total number of states (may not be unique)
        stCount = obs.shape[0]
        
        for st in range(stCount):
            targets[st] = rew[st] + (self.config.gamma * vals[st, 0])
            
        targets = np.array(targets).reshape((self.batch * int(float(self.env_.T) / float(self.env_.actionTime)), 1))
        
        avgMDPReturn = np.mean([p["returns"][0] for p in PATHS])

        self.finalPATHS = dict(
            observations = obs,
            actions = act,
            advantages = advn,
            returns = ret,
            targets = targets,
            avgmdp = avgMDPReturn
        )

        return self
    
    def computeProdProbability(self, event_maps):

        products = []

        for j in xrange(0, len(self.config.sharedSites)):
            owners = []
            locs = self.config.sharedSites[j]

            if not self.line.hasLocation(locs):
                continue

            for l in self.config.lines:
                if l.hasLocation(locs):
                    owners.extend(l.getOwners())

            for k2 in sorted(self.eventMap[locs].iterkeys()):
                prod = 1.0

                #print "Generating product of rewards for event with location ", locs, " and Time range ", k2, " - ", k2 - self.config.repeatKTimeSteps'
                for owner in owners:

                    if (owner == self.agent):
                        continue

                    ag_event_map = event_maps[owner]
                    #print float(ag_event_map[locs][k2]["val"])
                    #print (float(ag_event_map[locs][k2]["val"]) / float(self.batch))
                    prod *= (1.0 - (float(ag_event_map[locs][k2]["val"]) / float(self.batch)))
                    #print prod
                products.append(prod)
        return products
    
    def updateEventArrays(self, event_maps):
        
        # Current Constraint Number
        products = self.computeProdProbability(event_maps)
        self.products = products
        # print self.products

        eventPATHS = []

        # Constraints
        count = 0
        for j in xrange(0, len(self.config.sharedSites)):
            locs = self.config.sharedSites[j]

            if not self.line.hasLocation(locs):
                continue

            for k2 in sorted(self.eventMap[locs].iterkeys()):

                obs_lst = []
                act_lst = []
                prod = products[count]
                rew = self.constraints_rewards[count]
                
                # print "Generating for event with location ", locs, " and Time range ", k2, " - ", k2 - self.config.repeatKTimeSteps

                for k3 in sorted(self.eventMap[locs][k2].iterkeys()):
                    
                    # print k3, " ", self.eventMap[locs][k2][k3]
                    
                    if (k3 == "val"):
                        continue
                        
                    # Which batch to select.
                    samples = self.eventMap[locs][k2][k3]["sample"]
                    
                    # Till which time step to collect.
                    decision = self.eventMap[locs][k2][k3]["decision"]

                    for sample in samples:
                        # print "Adding for primitive event: ", k3

                        # Which batch to select.
                        sample = self.PATHS[sample]
                        
                        # Get all observation from the selected batch.
                        obs = sample["observations"]
                        # obs = np.array(obs).reshape((int(float(self.env_.T) / float(self.env_.actionTime)),self.env_.validLengthOfState))
                        
                        # Get all actions from the selected batch.
                        act = sample["actions"]

                        # Select all observations and actions upto the decision time.
                        obs_lst.extend(obs[0:decision+1])
                        act_lst.extend(act[0:decision+1])

                # print len(obs_lst)
                obs_lst = np.array(obs_lst, dtype=np.float32).reshape((len(obs_lst), self.env_.validLengthOfState))
                act_lst = np.array(act_lst, dtype=np.float32).reshape((len(act_lst), self.env_.numberOfActions))
                
                if len(obs_lst) != 0 or len(act_lst) != 0:
                    path = dict(
                        observations = obs_lst,
                        actions = act_lst,
                        constraint = count
                    )
                    # print path
                    eventPATHS.append(path)
                else:
                    pass
                    # print "Warning: ConsNumber: ", count, " agent: ", self.agent, " Locs: ", locs, " ", k2
                    
                # Next Constraint
                count += 1

        self.eventPATHS = eventPATHS
        return self
    
    # Returns formatted input to the NN.
    def giveMeEverything(self, pl):
        assert self.eventPATHS != None
        assert self.PATHS != None
        assert self.finalPATHS != None
        
        obs = self.finalPATHS["observations"]
        act = self.finalPATHS["actions"]
        
        advn = self.finalPATHS["advantages"]
                
        #!!!!!! Normalize whatever you want.
        advn = (advn - np.mean(advn)) / (np.std(advn) + 1e-8)
        
        tars = self.finalPATHS["targets"]

        cum_sum_array = [0]*(self.numberOfConstraints+1)
        cum_sum_array[0] = len(obs)

        cons_incl = [0.0]*self.numberOfConstraints

        input_given = []
        act_given = []
        input_given.extend(obs)
        act_given.extend(act)
        
        for i in xrange(0, len(self.eventPATHS)):
            dictionary = self.eventPATHS[i]
            obse = dictionary["observations"]
            acte = dictionary["actions"]
            cons_num = dictionary["constraint"]
            # print cons_num
            cum_sum_array[cons_num+1] = len(obse)
            cons_incl[cons_num] = 1.0
            input_given.extend(obse)
            act_given.extend(acte)
            
        # print cum_sum_array    
        for i in xrange(1, len(cum_sum_array)):
            if cum_sum_array[i] == 0:
                cum_sum_array[i] = cum_sum_array[i-1]
            else:
                cum_sum_array[i] += cum_sum_array[i-1]
        # print cum_sum_array
        # print cons_incl
        
        cons_incl = np.array(cons_incl, dtype=np.float32).reshape((self.numberOfConstraints, 1))
        cum_sum_array = np.array(cum_sum_array, dtype=np.int32).reshape((self.numberOfConstraints+1, 1))
        input_given = np.array(input_given).reshape((int(cum_sum_array[-1]), self.env_.validLengthOfState))
        act_given = np.array(act_given).reshape((int(cum_sum_array[-1]), self.env_.numberOfActions))
        train_mode_policy_given = True
        assert self.products != None
        assert self.constraints_rewards != None
        
        rew_scalars_given = np.array(self.constraints_rewards, dtype=np.float32).reshape((self.numberOfConstraints, 1))
        rew_scalars_given = (rew_scalars_given - np.mean(rew_scalars_given)) / (np.std(rew_scalars_given) + 1e-8)
        
        prod_scalars_given = np.array(self.products, dtype=np.float32).reshape((self.numberOfConstraints, 1))
        prod_scalars_given = (prod_scalars_given - np.mean(prod_scalars_given)) / (np.std(prod_scalars_given) + 1e-8)
        
        # print input_given.shape, act_given.shape
        
        feeding_dict = dict()
        feeding_dict[pl.arr_placeholder] = cum_sum_array
        feeding_dict[pl.constraint_included] = cons_incl
        feeding_dict[pl.action_var_policy] = act_given
        feeding_dict[pl.policy.train_mode] = train_mode_policy_given
        feeding_dict[pl.returns_var] = advn
        feeding_dict[pl.policy.input_var] = input_given
        feeding_dict[pl.rew_scalars] = rew_scalars_given
        feeding_dict[pl.prod_scalars] = prod_scalars_given
        
        baseline_dict = dict()
        baseline_dict[pl.input_var_baseline] = obs
        baseline_dict[pl.train_mode_baseline] = True
        baseline_dict[pl.targets] = tars
        
        testMap = self.fillTestMappingFunction(self.PATHS)
    
        return feeding_dict, baseline_dict, testMap
    
    def resetTestMappingFunction(self):

        testMap = dict()
        testMap["agent"] = self.agent
        testMap["MDP"] = self.finalPATHS["avgmdp"]

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

        # print json.dumps(testMap, indent=4, sort_keys=True)
        return testMap

    def fillTestMappingFunction(self, PATHS):
        testMap = self.resetTestMappingFunction()
        for ind in xrange(0, len(PATHS)):

            path = PATHS[ind]

            # print "=============Episode=========\n"

            observations = path["observations"]
            actions = path["actions"]
            nextObs = path["nextObs"]

            assert len(observations) == len(actions) == len(nextObs)
            length = len(observations)

            for j in xrange(0, length):
                obs = observations[j].tolist()
                next_obs = nextObs[j].tolist()
                act = actions[j]

                (location_obs, time_obs, inspection_obs, dold_obs, actualLocation_obs) = self.getDataForObs(obs)
                (location_nextobs, time_nextobs, inspection_nextobs, dold_nextobs, actualLocation_nextobs) = self.getDataForObs(next_obs)

                action_taken = self.env_.getAction(act)
                if action_taken == 0 and actualLocation_obs == actualLocation_nextobs and location_obs == location_nextobs and self.line.isShared(actualLocation_obs):

                    if (inspection_obs[location_obs] == 0 and dold_obs == 0):

                        if (inspection_nextobs[location_nextobs] == 1 and dold_nextobs == 0):

                            testMap[ind][actualLocation_obs][self.getTimeSlot(time_obs)] = True
        # print json.dumps(testMap, indent=4, sort_keys=True)
        return testMap


# In[3]:


# c = Config(200)

# print c.batch

# t = Train(0, c)
# t2 = Train(1, c)
# g_1 = tf.Graph()

# with g_1.as_default():
#     with tf.device('/device:GPU:2'):

#         # -----------------------------------------------------------------------------------------------------
#         # Baseline NN Computational Graph

#         input_var_baseline  = tf.placeholder(shape=[None, t.env_.validLengthOfState], dtype=tf.float32)
#         train_mode_baseline = tf.placeholder(tf.bool)
#         baseline = VFLearn(t.env_.validLengthOfState, input_var_baseline, train_mode_baseline, c)
        
#         input_var_policy  = tf.placeholder(shape=[None, t.env_.validLengthOfState], dtype=tf.float32)
#         train_mode_policy = tf.placeholder(tf.bool)
#         policy = ANNPol(t.env_.validLengthOfState, t.env_.numberOfActions, input_var_policy, train_mode_policy)
        
# with tf.Session(graph=g_1, config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
#     sess.run(tf.global_variables_initializer())
    
#     t.runIteration(policy, baseline)
#     t2.runIteration(policy, baseline)
    
#     eventMaps = []
#     eventMaps.append(t.eventMap)
#     eventMaps.append(t2.eventMap)
    
#     t = t.updateEventArrays(eventMaps)
#     t2 = t2.updateEventArrays(eventMaps)
#     assert t.eventPATHS != None
    
#     print t.giveMeEverything()

