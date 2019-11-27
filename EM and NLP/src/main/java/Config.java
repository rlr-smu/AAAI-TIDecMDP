import com.google.common.primitives.Ints;

import java.io.*;
import java.util.*;

/**
 * Created by tarun on 4/6/17.
 */
class Config {
    String solver = "bonmin";

    // Flag=1 implies reading Domain from files.
    int flag = 0;
    int experiment = 10383;
    int offset = 1000;
    int timeout = 1800;
    int GenRun = 1;
    static String workDir = "/home/tarun/IdeaProjects/EventDecMdp/";
    static String modelDir = "src/main/models/";
    double theta = 0.1;
    double gamma = 0.9999;
    double initialxval = 0.00001;
    double alpha = 0.9;
    double beta = 0.8;

    double deltaFinal = 1e-5;
    int noIterConvergenceCheck = 10;

    int agents;
    int nPrivatePerAgent;
    int nShared;
    int minSharing;
    int maxSharing;
    int minT;
    int maxT;
    int minTaction;
    int maxTaction;
    int agentMax;
    int nLocs;
    int[] auction;
    ArrayList<Integer> sharedSites;
    ArrayList<ArrayList<Integer> > locs;
    ArrayList<Integer> nloc;
    int[][] collectTimes;
    int [][][] transitTimes;
    int[] T;
    int[][] rewardCollection;
    int[] creward;
    int rmi = 5;
    int rma = 20;
    int Rmax;
    int Rmin;
    private Random r = new Random();

    Config(int experiment, int agents, int nPrivatePerAgent, int nShared, int minSharing, int maxSharing, int minT, int maxT, int minTaction, int maxTaction) {
        this.experiment = experiment;
        this.flag = 0;
        this.agents = agents;
        this.nPrivatePerAgent = nPrivatePerAgent;
        this.nShared = nShared;
        this.minSharing = minSharing;
        this.maxSharing = maxSharing;
        this.minT = minT;
        this.maxT = maxT;
        this.minTaction = minTaction;
        this.maxTaction = maxTaction;
        generateDomain();
    }

    Config(int experiment) {
        this.flag = 1;
        this.experiment = experiment;
        readConfig();
    }

    private void generateDomain() {
        agentMax = (int) Math.ceil((double) nShared * maxSharing / (double) agents);
        nLocs = (agents * nPrivatePerAgent) + nShared;
        auction = new int[nLocs];
        Arrays.fill(auction, -1);
        sharedSites = new ArrayList<>();
        locs = new ArrayList<>();
        nloc = new ArrayList<>();

        System.out.println("Experiment: " + experiment);
        System.out.println("Theta: " + theta);
        System.out.println("gamma: " + gamma);
        System.out.println("initialx: " + initialxval);
        System.out.println("alpha: " + alpha);
        System.out.println("beta: " + beta);
        System.out.println("deltaFinal: " + deltaFinal);
        System.out.println("Average Check for convergence: " + noIterConvergenceCheck);
        System.out.println("\nagents: " + agents);
        System.out.println("PrivatePer: " + nPrivatePerAgent);
        System.out.println("nShared: " + nShared);
        System.out.println("nLocs: " + nLocs);
        System.out.println("Agent Max: "+agentMax);

        for (int i = 0; i < agents; i++) {
            ArrayList<Integer> lst = new ArrayList<>();
            for (int j = 0; j < nPrivatePerAgent; j++) {
                int num = r.nextInt(nLocs);
                while (auction[num] != -1) {
                    num = r.nextInt(nLocs);
                }
                auction[num] = i;
                lst.add(num);
            }
            locs.add(lst);
        }

        class Store implements Comparable<Store> {
            int index;
            int count;

            Store(int index, int count) {
                this.index = index;
                this.count = count;
            }

            @Override
            public int compareTo(Store store) {
                if (this.count > store.count) return 1;
                else if (this.count < store.count) return -1;
                else return 0;
            }
        }

        Store[] stores = new Store[agents];
        for (int i = 0; i < agents ; i++) {
            stores[i] = new Store(i, agentMax);
        }
        Arrays.sort(stores, Collections.reverseOrder());

        for (int i = 0; i < nLocs; i++) {
            if (auction[i] != -1) {
                System.out.println("Location " + i + " Auctioned to: " + auction[i]);
                continue;
            }
            int tobesharedbetween = r.nextInt((maxSharing - minSharing) + 1) + minSharing;
            assert tobesharedbetween >= minSharing;
            assert tobesharedbetween <= maxSharing;
            Set<Integer> setOfAgents = new HashSet<>();
            for (int j = 0; j < tobesharedbetween ; j++) {
                Store s = stores[j];
                setOfAgents.add(s.index);
            }
            assert setOfAgents.size() == tobesharedbetween;

            for (int vals : setOfAgents) {
                for (Store s :  stores) {
                    if (s.index == vals) {
                        s.count -= 1;
                        break;
                    }
                }
                locs.get(vals).add(i);
            }
            Arrays.sort(stores, Collections.reverseOrder());

            auction[i] = -2;
            System.out.println("Location " + i + " Auctioned to: " + setOfAgents.toString());
            sharedSites.add(i);
        }

        for (ArrayList<Integer> arrs : locs) {
            nloc.add(arrs.size());
        }

        System.out.println("Auctioned: " + Arrays.toString(auction));
        System.out.println("AgentWise: " + locs.toString());
        System.out.println("SharedSites: " + sharedSites.toString());
        System.out.println("NoLocPerAgent: " + nloc.toString());

        collectTimes = new int[agents][];
        transitTimes = new int[agents][][];
        rewardCollection = new int[agents][];
        creward = new int[sharedSites.size()];

        int totalPow = r.nextInt((maxT - minT) + 1) + minT;
        T = new int[agents];
        Arrays.fill(T, (int) Math.pow(2, totalPow));
        for (int i = 0; i < agents; i++) {
            collectTimes[i] = new int[nloc.get(i)];
            rewardCollection[i] = new int[nloc.get(i)];
            transitTimes[i] = new int[nloc.get(i)][];
            int t = (int) Math.pow(2, r.nextInt((maxTaction - minTaction) + 1) + minTaction);
            Arrays.fill(collectTimes[i], t);

            for (int j = 0; j < transitTimes[i].length; j++) {
                transitTimes[i][j] = new int[nloc.get(i)];
                Arrays.fill(transitTimes[i][j], t);
                rewardCollection[i][j] = r.nextInt((rma - rmi) + 1) + rmi;
            }
        }

        for (int i = 0; i < creward.length; i++) {
            int crmi = (int)Math.ceil(2.0*rma);
            int crma = (int)Math.ceil(2.5*rma);
            creward[i] = r.nextInt((crma - crmi) + 1) + crmi;
        }

        System.out.println("TotalTime: " + Arrays.toString(T));
        System.out.println("Collect: " + Arrays.deepToString(collectTimes));
        System.out.println("Transit: " + Arrays.deepToString(transitTimes));
        System.out.println("MDPRew: " + Arrays.deepToString(rewardCollection));
        System.out.println("ConsReward: " + Arrays.toString(creward));

        Rmax = Ints.max(creward);
        Rmin = Ints.min(creward);
        for (int i = 0; i < agents ; i++) {
            int mm = Ints.max(rewardCollection[i]);
            int mm1 = Ints.min(rewardCollection[i]);
            if (mm > Rmax) {
                Rmax = mm;
            }
            if (mm1 < Rmin) {
                Rmin = mm1;
            }
        }

        System.out.println("Rmin: "+Rmin);
        System.out.println("Rmax: "+Rmax);
        writeConfig();
    }

    private void writeConfig() {
        try {
            FileOutputStream fos = new FileOutputStream(workDir+"Data/config"+experiment+".ser");
            ObjectOutputStream oos = new ObjectOutputStream(fos);
            oos.write(agents);
            oos.write(nPrivatePerAgent);
            oos.write(nShared);
            oos.write(nLocs);
            oos.writeObject(auction);
            oos.writeObject(locs);
            oos.writeObject(sharedSites);
            oos.writeObject(nloc);
            oos.writeObject(T);
            oos.writeObject(collectTimes);
            oos.writeObject(transitTimes);
            oos.writeObject(rewardCollection);
            oos.writeObject(creward);
            oos.write(Rmin);
            oos.write(Rmax);
            oos.close();
            fos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @SuppressWarnings("unchecked")
    private void readConfig() {
        try {
            FileInputStream fis = new FileInputStream(workDir+"Data/config"+experiment+".ser");
            ObjectInputStream oos = new ObjectInputStream(fis);
            agents = oos.read();
            nPrivatePerAgent = oos.read();
            nShared = oos.read();
            nLocs = oos.read();
            auction = (int[])oos.readObject();
            locs = (ArrayList<ArrayList<Integer>>) oos.readObject();
            sharedSites = (ArrayList<Integer>)oos.readObject();
            nloc = (ArrayList<Integer>)oos.readObject();
            T = (int[])oos.readObject();
            collectTimes = (int[][])oos.readObject();
            transitTimes = (int[][][]) oos.readObject();
            rewardCollection = (int[][])oos.readObject();
            creward = (int[])oos.readObject();
            Rmin = oos.read();
            Rmax = oos.read();
            oos.close();
            fis.close();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }

        System.out.println("Experiment: " + experiment);
        System.out.println("Theta: " + theta);
        System.out.println("gamma: " + gamma);
        System.out.println("initialx: " + initialxval);
        System.out.println("alpha: " + alpha);
        System.out.println("beta: " + beta);
        System.out.println("deltaFinal: " + deltaFinal);
        System.out.println("Average Check for convergence: " + noIterConvergenceCheck);
        System.out.println("\nagents: " + agents);
        System.out.println("PrivatePer: " + nPrivatePerAgent);
        System.out.println("nShared: " + nShared);
        System.out.println("nLocs: " + nLocs);
        System.out.println("Auctioned: " + Arrays.toString(auction));
        System.out.println("AgentWise: " + locs.toString());
        System.out.println("SharedSites: " + sharedSites.toString());
        System.out.println("NoLocPerAgent: " + nloc.toString());
        System.out.println("TotalTime: " + Arrays.toString(T));
        System.out.println("Collect: " + Arrays.deepToString(collectTimes));
        System.out.println("Transit: " + Arrays.deepToString(transitTimes));
        System.out.println("MDPRew: " + Arrays.deepToString(rewardCollection));
        System.out.println("ConsReward: " + Arrays.toString(creward));
        System.out.println("Rmin: "+Rmin);
        System.out.println("Rmax: "+Rmax);
    }
}
