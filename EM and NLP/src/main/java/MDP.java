import com.google.common.base.Stopwatch;
import org.apache.commons.lang3.StringUtils;

import java.io.*;
import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.TimeUnit;

import static java.lang.Math.pow;
import static java.lang.Math.random;

/**
 * Created by tarun on 4/6/17.
 */
public class MDP implements Callable<Void> {

    int T;
    ArrayList<Integer> locs;
    int nlocs;
    int agent;
    ArrayList<State> states;
    ArrayList<Action> actions;
    private int[] collectTimes;
    private int [][] transitTimes;
    double alpha;
    double beta;
    ArrayList<Double> start;
    ArrayList<String> binary;
    private Config config;
    State terminal;
    private int numberStates;
    int numberActions;
    int numberVariables;

    MDP(int agent, Config c) {
        this.T = c.T[agent];
        this.locs = c.locs.get(agent);
        this.nlocs = locs.size();
        this.agent = agent;
        this.binary = new ArrayList<>();
        generateBinaryStrings(this.nlocs);
        this.states = new ArrayList<>();
        this.actions = new ArrayList<>();
        this.collectTimes = c.collectTimes[agent];
        this.transitTimes = c.transitTimes[agent];
        this.alpha = c.alpha;
        this.beta = c.beta;
        this.start = new ArrayList<>();
        this.config = c;
    }

    @Override
    public Void call() {
        if (this.config.flag==0) {
            generateDomain();
        } else {
            deserializeDomain();
            defineStart();
            this.numberStates = states.size();
            this.numberActions = actions.size();
            this.numberVariables = numberStates * numberActions;
        }
        return null;
    }

    private void defineStart() {
        double sum = 0;
        for(State s : states) {
            boolean sameds = true;
            for (int i = 0; i < nlocs; i++) {
                if (s.dvals.charAt(i) != '0') {
                    sameds = false;
                    break;
                }
            }

            if(s.time==config.T[agent] && sameds && s.dold==0) {
                sum += 1;
            }
        }

        for(State s : states) {
            boolean sameds = true;
            for (int i = 0; i < nlocs; i++) {
                if (s.dvals.charAt(i) != '0') {
                    sameds = false;
                    break;
                }
            }
            if(s.time==config.T[agent] && sameds && s.dold==0) {
                start.add((double)1/sum);
            } else {
                start.add((double)0);
            }
        }


    }

    private void generateDomain() {
        String dum= StringUtils.repeat('2', nlocs);
        this.terminal = new State(0,-1,-1,-1, dum, -1, actions);
        this.states.add(this.terminal);
        this.initiateActions();
        this.initiateStates();
        //checkTransitionProbabilitySumTo1();
        this.waste();
        this.reindexStates();
        this.writeTransitions();
        this.writeRewards();
        this.serializeDomain();
        this.defineStart();
        this.numberStates = states.size();
        this.numberActions = actions.size();
        this.numberVariables = numberStates * numberActions;
    }

    ArrayList<double[]> generateLPAc() {
        System.out.println("Generating LP for "+agent);
        double Rmax = config.Rmax;
        double Rmin = config.Rmin;
        ArrayList<double[]> ret = new ArrayList<>();
        double[] R_mat = new double[numberVariables];
        double[] newR = new double[numberVariables];

        int store = 0;
        for(State s : states) {
            for(Reward rew : s.reward) {
                R_mat[store] = rew.reward;
                newR[store] = (rew.reward / (Rmax-Rmin));
                store += 1;
            }
        }

        ret.add(R_mat);
        ret.add(newR);
        return ret;
    }

    private void writeTransitions() {
        System.out.println("     Writing Transitions for Agent " +agent);
        for(Action a : actions) {
            for(State s : states) {
                for(State sd : states) {
                    double tt = transition(s,a,sd);
                    if (tt != 0) {
                        s.transition.add(new Transition(a.index, sd.index, tt));
                    }
                }
            }
        }
    }

    private void writeRewards() {
        System.out.println("     Writing Rewards for Agent " +agent);
        for(State s : states) {
            for(Action a : actions) {
                double reward = rewardFunction(s,a);
                s.reward.add(new Reward(a.index, reward));
            }
        }
    }

    private void reindexStates() {
        int index = 0;
        for (State s:states) {
            s.index = index;
            index += 1;
        }
    }

    private void serializeDomain() {
        try {
            FileOutputStream fos = new FileOutputStream(this.config.workDir + "Data/Domain" + agent + "_exp_" + config.experiment + ".ser");
            ObjectOutputStream oos = new ObjectOutputStream(fos);
            oos.writeObject(states);
            oos.writeObject(actions);
            for(State s: states) {
                oos.writeObject(s.transition);
                oos.writeObject(s.reward);
            }
            oos.flush();
            oos.close();
            fos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void deserializeDomain()    {
        try {
            FileInputStream fos = new FileInputStream(this.config.workDir + "Data/Domain" + agent + "_exp_" + config.experiment + ".ser");
            ObjectInputStream oos = new ObjectInputStream(fos);
            states = (ArrayList<State>) oos.readObject();
            terminal = states.get(0);
            actions = (ArrayList<Action>) oos.readObject();
            for(State s : states) {
                s.transition = (ArrayList<Transition>) oos.readObject();
                s.reward = (ArrayList<Reward>) oos.readObject();
            }
            oos.close();
            fos.close();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
    }

    void checkTransitionProbabilitySumTo1() {
        for (Action k : actions) {
            for (State i : states) {
                double sum = 0;
                for (State j : states) {
                    double tran = transition(i,k,j);
                    sum += tran;
                }
                if (sum!=1) {
                    System.out.println("WARNING: k: " + k + " i: " + i + " Sum: " + sum);
                }
            }
        }
    }

    private void generateBinaryStrings(int n) {
        for(int i = 0; i < pow(2,n); i++) {
            String form="%0"+n+"d";
            String bin = String.format(form, Integer.valueOf(Integer.toBinaryString(i)));
            binary.add(bin);
        }
    }

    private void initiateActions() {
        int index = 0;
        Action a = new Action(index, "Collect");
        this.actions.add(a);
        index += 1;
        for (int i = 0; i < nlocs ; i++) {
            a = new Action(index, "Go to Site "+i, i);
            index += 1;
            this.actions.add(a);
        }
    }

    private void initiateStates() {
        int index = 1;
        for (int i = 0; i <= this.T; i+= config.collectTimes[agent][0]) {
            for (int j = 0; j < nlocs ; j++) {
                for (String k:binary) {
                    ArrayList<Integer> lyst = new ArrayList<>();
                    if (k.charAt(j)=='0') {
                        lyst.add(0);
                    } else if (k.charAt(j)=='1') {
                        lyst.add(0);
                        lyst.add(1);
                    }
                    for (Integer t:lyst) {
                        State st = new State(index,j,locs.get(j),i,k,t,actions);
                        this.states.add(st);
                        index += 1;
                    }
                }
            }
        }
        System.out.println(states.size() * actions.size());
    }

    private double rewardFunction(State s, Action a) {
        if (s.equals(terminal)) {
            return 0.0;
        }
        int sdvalsloc = Character.getNumericValue(s.dvals.charAt(s.location));
        if(s.dold == 0 && sdvalsloc == 1) {
            return config.rewardCollection[agent][s.location] + random();
        } else {
            return 0.1 + random();
        }
    }

    double transition(State s, Action a, State sd) {
        if (a.name.equals("Collect")) {
            return transitionCollect(s,sd);
        } else {
            return transitionGoto(s,sd,a);
        }
    }

    private double transitionCollect(State s, State sd) {
        if (s.time == -1 && sd.equals(s)){
            return 1.0;
        }

        if (s.time==0 && sd.equals(terminal)) {
            return 1.0;
        }

        boolean sameds = true;
        for (int i = 0; i < nlocs; i++) {
            if (i != s.location) {
                if (s.dvals.charAt(i)!=sd.dvals.charAt(i)) {
                    sameds = false;
                    break;
                }
            }
        }

        if (!sameds) {
            return 0.0;
        }

        if (s.location != sd.location) {
            return 0.0;
        }

        if (sd.time != s.time-this.collectTimes[s.location]) {
            return 0.0;
        }

        if (sd.time < 0) {
            return 0.0;
        }

        int sdvalsloc = Character.getNumericValue(s.dvals.charAt(s.location));
        int sddvalsloc = Character.getNumericValue(sd.dvals.charAt(s.location));
        if (sd.dold != sdvalsloc) {
            return 0.0;
        }

        if (s.dold==1 && sdvalsloc==0) {
            return 0.0;
        }

        if (sdvalsloc==0) {
            return (sddvalsloc==1)?alpha:(1-alpha);
        } else if (sdvalsloc==1) {
            return (sddvalsloc==1)?1.0:0.0;
        } else {
            return 0.0;
        }
    }

    private double transitionGoto(State s, State sd, Action a) {
        int l = s.location;
        int ld = sd.location;
        int t = s.time;
        int td = sd.time;

        int dold = s.dold;
        int doldd = sd.dold;
        int dest = a.gotox;

        boolean sameds = true;
        for (int i = 0; i < nlocs; i++) {
            if (s.dvals.charAt(i) != sd.dvals.charAt(i)) {
                sameds = false;
                break;
            }
        }

        if (s.time==-1 && sd.equals(s)) {
            return 1.0;
        }

        if (s.time==0 && sd.equals(terminal)) {
            return 1.0;
        }

        if (!sameds) {
            return 0.0;
        }

        if ((ld != dest) && (ld != l)) {
            return 0.0;
        }

        if (td != t - transitTimes[l][ld]) {
            return 0.0;
        }

        if (td < 0) {
            return 0.0;
        }

        int sdvalsloc = Character.getNumericValue(s.dvals.charAt(s.location));
        int sddvalsloc = Character.getNumericValue(sd.dvals.charAt(sd.location));

        if (doldd != sddvalsloc) {
            return 0.0;
        }

        if (dold==1 && sdvalsloc==0) {
            return 0.0;
        }

        if (ld==l && ld==dest) {
            return 1.0;
        } else if (ld==l) {
            return 1 - beta;
        } else if (ld==dest) {
            return beta;
        } else {
            return 0.0;
        }
    }

    private ArrayList<State> removeWasteStates(int iter) {
        ArrayList<State> wasteStates  = new ArrayList<>();
        int sum = 0;
        double val = 0.0;
        Stopwatch timer = Stopwatch.createUnstarted();
        timer.start();
        int tots = states.size();
        int offset = config.offset;
        for (State s:states) {
            sum += 1;
            if (sum%offset == 0) {
                timer.stop();
                //System.out.println(timer);
                val = (double)timer.elapsed(TimeUnit.SECONDS);
                double ntimes = (double)sum / (double)offset;
                double avg = val / ntimes;
                double timerem = ((double)(tots-sum) / offset) * avg;
                System.out.println("["+agent+","+iter+"] Done. "+sum+" Out of: "+tots+ " Avg: "+avg + " Rem: "+timerem);
                timer.start();
            }

            if (s.equals(terminal)) {
                //System.out.println("sasassas");
                continue;
            }

            int num_value = Character.getNumericValue(s.dvals.charAt(s.location));
            if (s.dold==1 && num_value==0) {
                System.out.print("10 Anomaly");
                wasteStates.add(s);
                continue;
            }

            boolean sameds = true;
            for (int i = 0; i < nlocs; i++) {
                if (s.dvals.charAt(i) != '0') {
                    sameds = false;
                    break;
                }
            }

            if (s.time==T && sameds && s.dold==0) {
                continue;
            }

            int flag = 0;
            for(State sd : states) {
                for (Action a :  actions) {
                    //System.out.println(sd + "\n" + a + "\n" + s + "\n");
                    if (transition(sd,a,s)!=0) {
                        flag = 1;
                        break;
                    }
                }
                if (flag==1) {
                    break;
                }
            }

            if (flag==0) {
                wasteStates.add(s);
            }
        }
        timer.reset();
        states.removeAll(wasteStates);
        System.out.println("For agent " + agent+ " Iter "+iter+" done and removed "+wasteStates.size()+".");
        return wasteStates;
    }

    private ArrayList<State> succRemoval(ArrayList<State> removedSt, int iter) {
        ArrayList<State> wasteStates = new ArrayList<>();
        ArrayList<State> maybe = new ArrayList<>();
        wasteStates.clear();
        maybe.clear();

        int sum = 0;
        double val = 0.0;
        Stopwatch timer = Stopwatch.createUnstarted();
        timer.start();
        int tots = removedSt.size();
        int offset = config.offset;
        for (State rms:removedSt) {
            sum += 1;
            if (sum%offset == 0) {
                timer.stop();
                //System.out.println(timer);
                val = (double)timer.elapsed(TimeUnit.SECONDS);
                double ntimes = (double)sum / (double)offset;
                double avg = val / ntimes;
                double timerem;
                timerem = ((double)(tots-sum) / offset) * avg;
                System.out.println("["+agent+","+iter+"] Done. "+sum+" Out of: "+tots+ " Avg: "+avg + " Rem: "+timerem);
                timer.start();
            }

            for(State sts : states) {

                if (sts.equals(terminal)) {
                    continue;
                }

                int num_value = Character.getNumericValue(sts.dvals.charAt(sts.location));
                if (sts.dold==1 && num_value==0) {
                    System.out.print("10 Anomaly");
                    wasteStates.add(sts);
                    continue;
                }

                boolean sameds = true;
                for (int i = 0; i < nlocs; i++) {
                    if (sts.dvals.charAt(i) != '0') {
                        sameds = false;
                        break;
                    }
                }

                if (sts.time==T && sameds && sts.dold==0) {
                    continue;
                }

                for (Action a :  actions) {
                    if (transition(rms,a,sts)!=0) {
                        maybe.add(sts);
                        break;
                    }
                }
            }
        }

        for (State sts : maybe) {
            int flag = 0;
            for (State s : states) {
                for (Action a : actions) {
                    if (transition(s,a,sts)!=0.0) {
                        flag=1;
                        break;
                    }
                }
                if (flag==1) {
                    break;
                }
            }
            if (flag==0) {
                wasteStates.add(sts);
            }
        }

        ArrayList<State> prevs = new ArrayList<>();
        for (State torem : wasteStates) {
            if(states.contains(torem)) {
                prevs.add(torem);
                states.remove(torem);
            }
        }
        System.out.println("For agent " + agent+ " Iter "+iter+" done and removed "+prevs.size()+".");
        return prevs;
    }

    private void waste() {
        int iter = 1;
        ArrayList<State> removed = removeWasteStates(iter);
        while (removed.size() != 0) {
            iter += 1;
            removed = succRemoval(removed, iter);
        }
    }
}
