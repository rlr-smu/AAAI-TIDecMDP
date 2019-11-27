
# coding: utf-8

# In[1]:


from MDP import MDP
from PrimitiveEvent import PrimtiveEvent
from Event import Event
from Constraint import Constraint
from Env import Env
import multiprocessing.pool
import multiprocessing
import pp
import numpy as np
import time
import os
from NNPolicy import ANNPol
from ValueFunction import VFLearn
from PlaceholderClass import PlaceholderClass
from Train import Train
import copy_reg
import types
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathos.multiprocessing import ThreadingPool as Pool


# In[2]:


# os.system("taskset -p 0xFFFFFFFF %d" % os.getpid())

def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)

def _instance_method_alias_call(obj):
    return obj.call()

class Stopwatch:

    # Construct self and start it running.
    def __init__(self):
        self._creationTime = time.time()  # Creation time

    # Return the elapsed time since creation of self, in seconds.
    def elapsedTime(self):
        return time.time() - self._creationTime


# In[3]:


class EDECMDP:
    def __init__(self, config):
        self.config = config
        self.num_agents = config.agents
        self.mdps = None
        self.primitives = None
        self.events = None
        self.constraints = None
#         self.generateMDPs()
#         self.genPrimitiveEvents()
#         self.genEvents()
#         self.genConstraints()
#         self.genAMPL()
        
    def _instance_method_alias_call(self, obj):
        return obj.call()

    def generateMDPs(self):
        self.mdps = []
        results = []

        p = Pool(processes=(self.num_agents+2))

        for i in xrange(0, self.num_agents):
            print "Generating MDP for Agent" + str(i)
            a = MDP(i, self.config)
            self.mdps.append(a)
            
        res = p.amap(self._instance_method_alias_call, self.mdps)
        self.mdps = res.get()
        
        sum = 0
        for m in self.mdps:
            sum += m.numberVariables
        print "Total Number of Variables: ", sum

    def genPrimitiveEvents(self):
        print "Generating Primitive Events"
        self.primitives = []
        index = 0
        for q in xrange(0, self.num_agents):
            a = self.mdps[q]
            for z in a.line.getSharedLocations():
                for i in a.states:
                    if (i == a.terminal):
                        continue
                    iactual = a.line.getIndexOfLocation(z)
                    assert iactual != -1
                    if i.location == iactual and i.dvals[i.location] == 0 and i.dold == 0:
                        for k in a.states:
                            if (k == a.terminal):
                                continue
                            kactual = a.line.getIndexOfLocation(z)
                            assert kactual != -1
                            if k.location == kactual and k.dvals[k.location] == 1 and k.dold == 0:
                                if a.transition(i, a.actions[0], k) != 0:
                                    pe = PrimtiveEvent(q, i, a.actions[0], k, index)
                                    self.primitives.append(pe)
                                    index = index + 1

    def genEvents(self):
        print "Generating Events"
        self.events = []
        index = 0
        for agent in xrange(0, self.num_agents):
            m = self.mdps[agent]
            for loc in m.line.getSharedLocations():
                j = self.config.T
                while j >= 0:
                    start = j
                    if (j - self.config.repeatKTimeSteps) >= 0:
                        end = j - self.config.repeatKTimeSteps
                    else:
                        end = 0

                    arr = []
                    for p in self.primitives:
                        sactual = m.line.getIndexOfLocation(loc)
                        assert sactual != -1
                        if p.agent == agent and p.state.location == sactual and p.state.time <= start and p.state.time > end:
                            arr.append(p)

                    if (len(arr) >= 1):
                        e = Event(agent, arr, index, "Agent " + str(agent) + " start inspection at site " + str(loc) + " between time steps " + str(start) + " - " + str(end), loc, start, end)
                        index = index + 1
                        self.events.append(e)
                    j -= self.config.repeatKTimeSteps

    def genConstraints(self):
        print "Generating Constraints"
        self.constraints = []
        index = 0
        shared = self.config.sharedSites
        for x in xrange(0, len(shared)):
            loc = shared[x]

            owners = []
            for line in self.config.lines:
                if line.isShared(loc) != False:
                    owners.extend(line.getOwners())

            j = self.config.T
            while j >=0 :
                start = j
                if (j - self.config.repeatKTimeSteps) >= 0:
                    end = j - self.config.repeatKTimeSteps
                else:
                    end = 0

                arr = []
                for e in self.events:
                    if (e.agent in owners and e.site == loc and e.startTime <= start and e.startTimeEnd >= end):
                        arr.append(e)

                if (len(arr) >= 1):
                    c = Constraint(arr, self.config.creward[x], index, "At least one agent inspect " + str(loc) + " between " + str(start) + " - " + str(end))
                    index += 1
                    self.constraints.append(c)

                j -= self.config.repeatKTimeSteps
                print

        # for c in self.constraints:
        #     print c.name
        #     print c.reward
        #
        #     for e in c.Events:
        #         print e.name
        #
        #         for p in e.pevents:
        #             print p.index
        #             print p.state
        #             print p.action
        #             print p.statedash
        #
        #         print
        #     print

    def getAvgRewardFromEvents(self, test_maps):
        batch_size = self.config.batch
        mdp_avg = 0.0
        event_avg = 0.0
        system_avg = 0.0
        all_mdp_vals = []

        for map_val in test_maps:
            all_mdp_vals.append(map_val["MDP"])
            mdp_avg += map_val["MDP"]

        system_avg += mdp_avg

        batch_rewards = []
        for batch in xrange(0, batch_size):
            #print "==============++Episode==================", batch
            #print

            batch_reward = 0
            
            # For each shared site.
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

                    #print "Generating for event with location ", loc, " and Time range ", start, " - ", start - self.config.repeatKTimeSteps

                    for owner in owners:
                        map_vals = test_maps[owner]
                        if map_vals[batch][loc][start] == True:
                            #print "Happened: event with location ", loc, " and Time range ", start, " - ", start - self.config.repeatKTimeSteps, " by: ", owner
                            #print
                            batch_reward += self.config.creward[lindex]
                            break

            batch_rewards.append(batch_reward)
        batch_rewards = np.array(batch_rewards)
        event_avg = np.mean(batch_rewards)
        system_avg += event_avg
        return mdp_avg, event_avg, system_avg, all_mdp_vals

    def genAMPL(self):
        print "     Generating AMPL: ",
        ampl = open(self.config.workDir + 'Data/nl2_exp_' + str(self.config.experiment) + '.dat', 'w')
        ampl.write("param n := " + str(self.num_agents) + ";\n")
        ampl.write("\n")
        for i in xrange(0, self.num_agents):
            ampl.write("set S[" + str(i + 1) + "] := ")
            for x in self.mdps[i].states:
                ampl.write(str(x.index + 1) + " ")
            ampl.write(";\n")
            ampl.write("set A[" + str(i + 1) + "] := ")
            for x in self.mdps[i].actions:
                ampl.write(str(x.index + 1) + " ")
            ampl.write(";\n")
        ampl.write("\n")
        ampl.write("param numcons := " + str(len(self.constraints)) + ";\n")
        ampl.write("param numprims := " + str(len(self.primitives)) + ";\n")
        ampl.write("param numevents := " + str(len(self.events)) + ";\n")
        ampl.write("param gamma := " + str(self.config.gamma) + ";\n")
        ampl.write("\n")

        ampl.write("param: sparseP: sparsePVal:= \n")
        for i in xrange(0, self.num_agents):
            for j in xrange(0, len(self.mdps[i].actions)):
                for k in xrange(0, len(self.mdps[i].states)):
                    h = self.mdps[i].states[k].transition
                    hh = [(x[1], x[2]) for x in h if x[0] == self.mdps[i].actions[j].index and x[2] != 0]
                    for valsac in hh:
                        ampl.write(str(i + 1) + " " + str(self.mdps[i].actions[j].index + 1) + " " + str(
                            self.mdps[i].states[k].index + 1) + " " + str(valsac[0] + 1) + " " + str(valsac[1]) + "\n")
            if i == self.num_agents - 1:
                ampl.write(";")
        ampl.write("\n")

        ampl.write("param R := \n")
        for i in xrange(0, self.num_agents):
            ampl.write("[" + str(i + 1) + ",*,*] : ")
            for j in xrange(0, len(self.mdps[i].actions)):
                ampl.write(str(j + 1) + " ")
            ampl.write(":= \n")
            for j in xrange(0, len(self.mdps[i].states)):
                ampl.write(str(j + 1) + " ")
                h = self.mdps[i].states[j].reward
                hh = [x[1] for x in h]
                for g in hh:
                    ampl.write(str(g) + " ")
                ampl.write("\n")
            if i == self.num_agents - 1:
                ampl.write(";")
        ampl.write("\n")

        ampl.write("param alpha := \n")
        for i in xrange(0, self.num_agents):
            ampl.write("[" + str(i + 1) + ",*] := ")
            for gg in xrange(0, len(self.mdps[i].start)):
                ampl.write(str(gg + 1) + " " + str(self.mdps[i].start[gg]) + " ")
            ampl.write("\n")
        ampl.write(";\n")

        ampl.write("param creward := ")
        for x in xrange(0, len(self.constraints)):
            ampl.write(str(x + 1) + " " + str(self.constraints[x].reward) + " ")
        ampl.write(";\n")

        ampl.write("param primitives : ")
        for i in xrange(0, 4):
            ampl.write(str(i + 1) + " ")
        ampl.write(":= \n")
        for z in self.primitives:
            ampl.write(str(z.index + 1) + " " + str(z.agent + 1) + " " + str(z.state.index + 1) + " " +
                       str(z.action.index + 1) + " " + str(z.statedash.index + 1) + "\n")
        ampl.write(";\n")

        for i in xrange(0, len(self.events)):
            ampl.write("set events[" + str(i + 1) + "] := ")
            for x in self.events[i].pevents:
                ampl.write(str(x.index + 1) + " ")
            ampl.write(";\n")
        ampl.write("\n")

        for i in xrange(0, len(self.constraints)):
            ampl.write("set cons[" + str(i + 1) + "] := ")
            for x in self.constraints[i].Events:
                ampl.write(str(x.index + 1) + " ")
            ampl.write(";\n")
        ampl.write("\n")

        ampl.write("for {(i,j,k,l) in sparseP} {	let P[i,j,k,l] := sparsePVal[i,j,k,l]; }")
        ampl.close()
        print "Done"
        
    def _instance_method_alias_run_iteration(self, obj, session):
        return obj.runIteration(session)
    
    def _instance_method_alias_update_eventArrays(self, obj, event_maps):
        return obj.updateEventArrays(event_maps)
    
    def merge_two_dicts(self, x, y):
        z = x.copy()   # start with x's keys and values
        z.update(y)    # modifies z with y's keys and values & returns None
        return z
   
    def initializeRL(self, filename):

        f = open(self.config.workDir+'Logs/'+filename+'.csv', 'w', 0)
        f.write('Iteration,AvgMDPReward,AvgEventReward,AvgSystemReward,KMDPReward,KEventReward,KSystemReward,Time\n')
        
        b = open(self.config.workDir+'Logs/Baseline_'+filename+'.csv', 'w', 0)
        b.write('Iteration,')
        for j in xrange(0, self.num_agents):
            b.write('Agent_'+str(j)+'_Initial Loss,'+'Agent_'+str(j)+'_Final Loss, ,')
        b.write('\n')
        
        pol = open(self.config.workDir+'Logs/Policy_'+filename+'.csv', 'w', 0)
        pol.write('Iteration,')
        for j in xrange(0, self.num_agents):
            pol.write('Agent_'+str(j)+'_Initial Loss,'+'Agent_'+str(j)+'_Final Loss, ,')
        pol.write('\n')

        trains = []        
        event_maps = []
        test_maps = []
        files = []
        placeholders = []

        for i in xrange(0, self.num_agents):
            train_instance = Train(i, self.config)
            event_maps.append(train_instance.eventMap)
            trains.append(train_instance)
            
        g_1 = tf.Graph()

        with g_1.as_default():
            for i in xrange(0, self.num_agents):
                with tf.device('/device:CPU:1'+str(i)):
                    pl = PlaceholderClass(agent=i, c=self.config, t=trains[i])
                    pl = pl.computationalGraphs()
                    trains[i].policy = pl.policy
                    trains[i].baseline = pl.baseline
                    assert pl.t.policy != None
                    assert pl.t.baseline != None
                    placeholders.append(pl)
                    

        with tf.Session(graph=g_1, config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            
            if self.config.runWithSavedModel:
                saver.restore(sess, "../models/model_"+filename+".ckpt")
                print("Model restored.")

            avg_arr = []
            ev_avg_arr = []
            mdp_avg_arr = []

            p = Pool(processes=(self.num_agents+2))
            stopwatch = Stopwatch()
            
            orig_base_loss = [placeholders[j].loss_baseline for j in xrange(0, self.num_agents)]
            learn_baseline = [placeholders[j].learning_step_baseline for j in xrange(0, self.num_agents)]
            new_base_loss = [placeholders[j].loss_baseline for j in xrange(0, self.num_agents)]
            
            orig_policy_loss = [placeholders[j].loss for j in xrange(0, self.num_agents)]
            learn_policy = [placeholders[j].learning_step for j in xrange(0, self.num_agents)]
            new_policy_loss = [placeholders[j].loss for j in xrange(0, self.num_agents)]

            writer = tf.summary.FileWriter('../tflogs/'+filename, sess.graph)
            
            for curr_iter in xrange(1, self.config.numIterations+1):
#                 print "---------Current Iteration----------: ", curr_iter

                event_maps = []
                test_maps = []
 
                res = p.amap(self._instance_method_alias_run_iteration, trains, [sess]*self.num_agents)
                trains = res.get()
                
                for j in xrange(0, self.num_agents):
                    event_maps.append(trains[j].eventMap)
                
                res = p.amap(self._instance_method_alias_update_eventArrays, trains, [event_maps]*self.num_agents)
                trains = res.get()
                
                feeding_dicts = []
                baseline_dicts = []
                test_maps = []
                
                for j in xrange(0, self.num_agents):
                    feeding_d, baseline_d, testMap_d = trains[j].giveMeEverything(placeholders[j])                    
                    feeding_dicts.append(feeding_d)
                    baseline_dicts.append(baseline_d)
                    test_maps.append(testMap_d)
                
                assert self.num_agents >= 2
                merged_dicts_baseline = self.merge_two_dicts(baseline_dicts[0], baseline_dicts[1])
                merged_dicts_feeding = self.merge_two_dicts(feeding_dicts[0], feeding_dicts[1])
                for j in xrange(2, self.num_agents):
                    merged_dicts_baseline = self.merge_two_dicts(merged_dicts_baseline, baseline_dicts[j])
                    merged_dicts_feeding = self.merge_two_dicts(merged_dicts_feeding, feeding_dicts[j])
                
                #-----------------------
                # Baseline training
                #-----------------------
                
                for j in xrange(0, self.num_agents):
                    # Set training mode to False for evaluating loss.
                    merged_dicts_baseline[placeholders[j].train_mode_baseline] = False
#                 print "Original Baseline Loss: ",
                orig_base_loss_vals = sess.run(orig_base_loss, feed_dict=merged_dicts_baseline)
#                 print orig_base_loss_vals

                for j in xrange(0, self.num_agents):
                    # Set training mode to True for learning step.
                    merged_dicts_baseline[placeholders[j].train_mode_baseline] = True
                sess.run(learn_baseline, feed_dict=merged_dicts_baseline)
                
                for j in xrange(0, self.num_agents):
                    # Set training mode to False for evaluating loss after training.
                    merged_dicts_baseline[placeholders[j].train_mode_baseline] = False
#                 print "New Baseline Loss: ", 
                new_base_loss_vals = sess.run(new_base_loss, feed_dict=merged_dicts_baseline)
#                 print new_base_loss_vals
                
                #-----------------------
                # Policy training
                #-----------------------

                for j in xrange(0, self.num_agents):
                    # Set training mode to False for evaluating loss.
                    merged_dicts_feeding[placeholders[j].policy.train_mode] = False
                orig_policy_loss_vals = sess.run(orig_policy_loss, feed_dict=merged_dicts_feeding)
#                 print "Original Policy Loss: ", orig_policy_loss_vals

                for j in xrange(0, self.num_agents):
                    # Set training mode to True for learning step.
                    merged_dicts_feeding[placeholders[j].policy.train_mode] = True
                sess.run(learn_policy, feed_dict=merged_dicts_feeding)

                for j in xrange(0, self.num_agents):
                    # Set training mode to False for evaluating loss after training.
                    merged_dicts_feeding[placeholders[j].policy.train_mode] = False
                new_policy_loss_vals = sess.run(new_policy_loss, feed_dict=merged_dicts_feeding)
#                 print "New Policy Loss: ", new_policy_loss_vals
                
                mdp_avg_val, event_avg_val, system_avg_val, all_mdp_values = self.getAvgRewardFromEvents(test_maps)

#                 print '\tIT:', curr_iter, '  Individual Average MDP Return:', all_mdp_values
#                 print '\tIT:', curr_iter, '  Average MDP Return:', mdp_avg_val
#                 print '\tIT:', curr_iter, '  Average Event Return:', event_avg_val
#                 print '\tIT:', curr_iter, '  Average Return:', system_avg_val

                mdp_avg_arr.append(mdp_avg_val)
                mdp_avg_k_val = np.mean(np.array(mdp_avg_arr[-100:]))
#                 print '\tLast K iter MDP avg ', mdp_avg_k_val, '\n'

                ev_avg_arr.append(event_avg_val)
                ev_avg_k_val = np.mean(np.array(ev_avg_arr[-100:]))
#                 print '\tLast K iter Event avg ', ev_avg_k_val, '\n'

                avg_arr.append(system_avg_val)
                avg_arr_k_val = np.mean(np.array(avg_arr[-100:]))
#                 print '\tLast K iter System avg ', avg_arr_k_val, '\n'
                
                elapTime = stopwatch.elapsedTime()
#                 print '\tElapsed Time: ', elapTime
                avg_time_per_iter = elapTime/curr_iter
#                 print '\tAverage Time per Iteration: ', avg_time_per_iter
                
                if curr_iter % self.config.savingThreshold == 0:
                    save_path = saver.save(sess, "../models/model_"+filename+".ckpt")
                    print("Model saved in path: %s" % save_path)
                
                if curr_iter >= self.config.loggingThreshold:
                    summary = tf.Summary()               
                    summary.value.add(tag='Average/mdp_avg', simple_value = mdp_avg_val)
                    summary.value.add(tag='Average/event_avg', simple_value = event_avg_val)
                    summary.value.add(tag='Average/system_avg', simple_value = system_avg_val)
                    summary.value.add(tag='KAverage/mdp_avg_k', simple_value = mdp_avg_k_val)
                    summary.value.add(tag='KAverage/event_avg_k', simple_value = ev_avg_k_val)
                    summary.value.add(tag='KAverage/system_avg_k', simple_value = avg_arr_k_val)
                    summary.value.add(tag='Timing/elapsed_time_var', simple_value = elapTime)
                    summary.value.add(tag='Timing/avg_time_per_iteration', simple_value = avg_time_per_iter)
                    writer.add_summary(summary, curr_iter)
                    writer.flush()

                    f.write(str(curr_iter)+','+str(mdp_avg_val)+','+str(event_avg_val)+','+str(system_avg_val)+',')
                    f.write(str(mdp_avg_k_val) + ',')
                    f.write(str(ev_avg_k_val) + ',')
                    f.write(str(avg_arr_k_val) + ',')
                    f.write(str(elapTime)+'\n')
                    b.write(str(curr_iter)+",")
                    pol.write(str(curr_iter)+",")
                    for j in xrange(0, self.num_agents):
                        b.write(str(orig_base_loss_vals[j])+","+str(new_base_loss_vals[j])+", ,")
                        pol.write(str(orig_policy_loss_vals[j])+","+str(new_policy_loss_vals[j])+", ,")
                    b.write("\n")
                    pol.write("\n")
                
        p.close()
        f.close()
        b.close()
        pol.close()
        

