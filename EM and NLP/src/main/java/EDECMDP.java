import com.ampl.AMPL;
import com.ampl.AMPLException;
import com.ampl.DataFrame;
import com.ampl.Variable;
import com.google.common.base.Stopwatch;

import java.io.*;
import java.text.DecimalFormat;
import java.util.*;
import java.util.concurrent.*;

/**
 * Created by tarun on 5/6/17.
 */
public class EDECMDP {
    private static final String ANSI_RESET = "\u001B[0m";
    private static final String ANSI_BLACK = "\u001B[30m";
    private static final String ANSI_RED = "\u001B[31m";
    private static final String ANSI_GREEN = "\u001B[32m";
    private static final String ANSI_YELLOW = "\u001B[33m";
    private static final String ANSI_BLUE = "\u001B[34m";
    private static final String ANSI_PURPLE = "\u001B[35m";
    private static final String ANSI_CYAN = "\u001B[36m";
    private static final String ANSI_WHITE = "\u001B[37m";

    private int num_agents;
    int number_of_variables;
    private ArrayList<MDP> mdps;
    private ArrayList<PrimtiveEvent> primitives;
    private ArrayList<Event> events;
    private ArrayList<Constraint> constraints;
    private Config config;
    private AMPL[] ampls;
    private ArrayList<double[]> prev_x;

    EDECMDP(Config c) {
        this.config = c;
        this.num_agents = config.agents;
        mdps = new ArrayList<>();
        primitives = new ArrayList<>();
        events = new ArrayList<>();
        constraints = new ArrayList<>();
        ampls = new AMPL[this.num_agents];
        generateMDPs();
        genPrimitiveEvents();
        genEvents();
        genConstraints();
    }

    private String getFreeSystemMemory() {
        Process p = null;
        try {
            p = Runtime.getRuntime().exec(new String[]{"sh", "-c", "free -m -b | grep \"Mem:\" | awk '{print $4}'"});
            BufferedReader stdInput = new BufferedReader(new InputStreamReader(p.getInputStream()));
            String s = stdInput.readLine();
            Long mem = Long.parseLong(s);
            return Driver.humanReadableByteCount(mem, false);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return "";
    }

    private void generateMDPs() {
        List<Callable<Void>> taskList = new ArrayList<>();
        for (int i = 0; i < num_agents; i++) {
            MDP mdp = new MDP(i, config);
            mdps.add(mdp);
            taskList.add(mdp);
        }
        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        try {
            List<Future<Void>> futureList = executor.invokeAll(taskList);
            for (Future<Void> voidFuture : futureList) {
                try {
                    voidFuture.get();
                } catch (ExecutionException e) {
                    System.err.println("Error executing task " + e.getMessage());
                    e.printStackTrace();
                }
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        executor.shutdownNow();
        int sum = 0;
        for (MDP m : mdps) {
            sum += m.numberVariables;
        }
        number_of_variables = sum;
        System.out.println("Total Number of Variables: " + sum);
    }

    private void genPrimitiveEvents() {
        System.out.println("Generating Primitive Events");
        int index = 0;
        for (int q = 0; q < num_agents; q++) {
            MDP a = mdps.get(q);
            for (Integer z : a.locs) {
                for (State i : a.states) {
                    if (i.equals(a.terminal)) {
                        continue;
                    }
                    int idvalsilocation = Character.getNumericValue(i.dvals.charAt(i.location));
                    if (i.actualLocation == z && idvalsilocation == 0 && i.dold == 0) {
                        for (State k : a.states) {
                            if (k.equals(a.terminal)) {
                                continue;
                            }
                            int kdvalsklocation = Character.getNumericValue(k.dvals.charAt(k.location));
                            if (k.actualLocation == z && kdvalsklocation == 1 && k.dold == 0) {
                                if (a.transition(i, a.actions.get(0), k) != 0) {
                                    PrimtiveEvent p = new PrimtiveEvent(q, i, a.actions.get(0), k, index);
                                    primitives.add(p);
                                    index += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private void genEvents() {
        System.out.println("Generating Events");
        int index = 0;
        for (int agent = 0; agent < num_agents; agent++) {
            for (Integer j : mdps.get(agent).locs) {
                ArrayList<PrimtiveEvent> arr = new ArrayList<>();
                for (PrimtiveEvent i : primitives) {
                    if (i.agent == agent && i.state.actualLocation == j) {
                        arr.add(i);
                    }
                }
                Event e = new Event(agent, arr, index, "Agent " + agent + " Collect at site " + j, j);
                index += 1;
                events.add(e);
            }
        }
    }

    private void genConstraints() {
        System.out.println("Generating Constraints");
        int index = 0;
        for (int x = 0; x < config.sharedSites.size(); x++) {
            ArrayList<Event> local = new ArrayList<>();
            for (Event y : events) {
                if (y.site == config.sharedSites.get(x)) {
                    local.add(y);
                }
            }
            Constraint c = new Constraint(local, config.creward[x], index);
            index += 1;
            constraints.add(c);
        }
    }

    private void genAMPL() {
        System.out.print("     Generating AMPL: ");
        String FILENAME = Config.workDir + "Data/nl2_exp_" + config.experiment + ".dat";
        BufferedWriter bw = null;
        FileWriter fw = null;
        try {
            fw = new FileWriter(FILENAME);
            bw = new BufferedWriter(fw);
            bw.write("param n := " + num_agents + ";\n");
            for (int i = 0; i < num_agents; i++) {
                bw.write("set S[" + (i + 1) + "] := ");
                for (State x : mdps.get(i).states) {
                    bw.write(x.index + 1 + " ");
                }
                bw.write(";\n");
                bw.write("set A[" + (i + 1) + "] := ");
                for (Action a : mdps.get(i).actions) {
                    bw.write(a.index + 1 + " ");
                }
                bw.write(";\n");
            }
            bw.write("\n");
            bw.write("param numcons := " + constraints.size() + ";\n");
            bw.write("param numprims := " + primitives.size() + ";\n");
            bw.write("param numevents := " + events.size() + ";\n");
            bw.write("param gamma := " + config.gamma + ";\n");
            bw.write("\n");

            bw.write("param: sparseP: sparsePVal:= \n");
            for (int i = 0; i < num_agents; i++) {
                for (State s : mdps.get(i).states) {
                    ArrayList<Transition> h = s.transition;
                    for (Transition t : h) {
                        bw.write((i + 1) + " " + (t.action_index + 1) + " " + (s.index + 1) + " " +
                                (t.statedash_index + 1) + " " + new DecimalFormat("#.#").format(t.probability) + "\n");
                    }
                }
                if (i == num_agents - 1) {
                    bw.write(";\n");
                }
            }
            bw.write("\n");

            bw.write("param R := \n");
            for (int i = 0; i < num_agents; i++) {
                bw.write("[" + (i + 1) + ",*,*] : ");
                for (int j = 0; j < mdps.get(i).actions.size(); j++) {
                    bw.write((j + 1) + " ");
                }
                bw.write(":=\n");
                for (State s : mdps.get(i).states) {
                    bw.write((s.index + 1) + " ");
                    ArrayList<Reward> h = s.reward;
                    for (Reward r : h) {
                        bw.write(r.reward + " ");
                    }
                    bw.write("\n");
                }
                if (i == num_agents - 1) {
                    bw.write(";\n");
                }
                bw.write("\n");
            }
            bw.write("\n");
            bw.write("param alpha := \n");
            for (int i = 0; i < num_agents; i++) {
                bw.write("[" + (i + 1) + ",*] := ");
                for (int j = 0; j < mdps.get(i).start.size(); j++) {
                    bw.write((j + 1) + " " + (mdps.get(i).start.get(j)) + " ");
                }
                bw.write("\n");
            }
            bw.write(";\n");
            bw.write("\n");

            bw.write("param creward := ");
            for (int i = 0; i < constraints.size(); i++) {
                bw.write((i + 1) + " " + (constraints.get(i).reward) + " ");
            }
            bw.write(";\n");
            bw.write("param primitives : ");
            for (int i = 0; i < 4; i++) {
                bw.write((i + 1) + " ");
            }
            bw.write(":= \n");
            for (PrimtiveEvent p : primitives) {
                bw.write((p.index + 1) + " " + (p.agent + 1) + " " + (p.state.index + 1) +
                        " " + (p.action.index + 1) + " " + (p.statedash.index + 1) + "\n");
            }
            bw.write(";\n");
            for (int i = 0; i < events.size(); i++) {
                bw.write("set events[" + (i + 1) + "] := ");
                for (PrimtiveEvent p : events.get(i).pevents) {
                    bw.write((p.index + 1) + " ");
                }
                bw.write(";\n");
            }
            bw.write("\n");

            for (int i = 0; i < constraints.size(); i++) {
                bw.write("set cons[" + (i + 1) + "] := ");
                for (Event e : constraints.get(i).events) {
                    bw.write((e.index + 1) + " ");
                }
                bw.write(";\n");
            }
            bw.write("\n");

            bw.write("for {(i,j,k,l) in sparseP} {	let P[i,j,k,l] := sparsePVal[i,j,k,l]; }");
            System.out.println("Done");
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (bw != null) {
                    bw.close();
                }
                if (fw != null) {
                    fw.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private void genAMPLSingle(int agent) {
        System.out.print("     Generating AMPL for Agent " + (agent) + " : ");
        String FILENAME = Config.workDir + "Data/single" + (agent) + "_exp_" + (config.experiment) + ".dat";
        BufferedWriter bw = null;
        FileWriter fw = null;
        try {
            fw = new FileWriter(FILENAME);
            bw = new BufferedWriter(fw);
            bw.write("param n := " + num_agents + ";\n");
            bw.write("param agent := " + (agent + 1) + ";\n");
            bw.write("param gamma := " + config.gamma + ";\n");
            for (int i = 0; i < num_agents; i++) {
                bw.write("set S[" + (i + 1) + "] := ");
                for (State x : mdps.get(i).states) {
                    bw.write(x.index + 1 + " ");
                }
                bw.write(";\n");
                bw.write("set A[" + (i + 1) + "] := ");
                for (Action a : mdps.get(i).actions) {
                    bw.write(a.index + 1 + " ");
                }
                bw.write(";\n");
            }
            bw.write("\n");

            bw.write("param: sparseP: sparsePVal:= \n");
            for (int i = 0; i < num_agents; i++) {
                for (State s : mdps.get(i).states) {
                    ArrayList<Transition> h = s.transition;
                    for (Transition t : h) {
                        bw.write((i + 1) + " " + (t.action_index + 1) + " " + (s.index + 1) + " " +
                                (t.statedash_index + 1) + " " + new DecimalFormat("#.#").format(t.probability) + "\n");
                    }
                }
                if (i == num_agents - 1) {
                    bw.write(";\n");
                }
            }
            bw.write("\n");

            bw.write("param R := \n");
            for (int i = 0; i < num_agents; i++) {
                bw.write("[" + (i + 1) + ",*,*] : ");
                for (int j = 0; j < mdps.get(i).actions.size(); j++) {
                    bw.write((j + 1) + " ");
                }
                bw.write(":=\n");
                for (State s : mdps.get(i).states) {
                    bw.write((s.index + 1) + " ");
                    ArrayList<Reward> h = s.reward;
                    for (Reward r : h) {
                        bw.write(r.reward + " ");
                    }
                    bw.write("\n");
                }
                if (i == num_agents - 1) {
                    bw.write(";\n");
                }
                bw.write("\n");
            }
            bw.write("\n");

            bw.write("param alpha := \n");
            for (int i = 0; i < num_agents; i++) {
                bw.write("[" + (i + 1) + ",*] := ");
                for (int j = 0; j < mdps.get(i).start.size(); j++) {
                    bw.write((j + 1) + " " + (mdps.get(i).start.get(j)) + " ");
                }
                bw.write("\n");
            }
            bw.write(";\n");
            bw.write("\n");

            bw.write("set all_numcons := ");
            for (int i = 0; i < constraints.size(); i++) {
                bw.write((i + 1) + " ");
            }
            bw.write(";\n");

            bw.write("set all_numprims := ");
            for (int i = 0; i < primitives.size(); i++) {
                bw.write((i + 1) + " ");
            }
            bw.write(";\n");

            bw.write("set all_numevents := ");
            for (int i = 0; i < events.size(); i++) {
                bw.write((i + 1) + " ");
            }
            bw.write(";\n");

            bw.write("param all_creward := ");
            for (int i = 0; i < constraints.size(); i++) {
                bw.write((i + 1) + " " + (constraints.get(i).reward) + " ");
            }
            bw.write(";\n");

            bw.write("param all_primitives : ");
            for (int i = 0; i < 4; i++) {
                bw.write((i + 1) + " ");
            }
            bw.write(":= \n");
            for (PrimtiveEvent p : primitives) {
                bw.write((p.index + 1) + " " + (p.agent + 1) + " " + (p.state.index + 1) +
                        " " + (p.action.index + 1) + " " + (p.statedash.index + 1) + "\n");
            }
            bw.write(";\n");

            for (int i = 0; i < events.size(); i++) {
                bw.write("set all_events[" + (i + 1) + "] := ");
                for (PrimtiveEvent p : events.get(i).pevents) {
                    bw.write((p.index + 1) + " ");
                }
                bw.write(";\n");
            }
            bw.write("\n");

            for (int i = 0; i < constraints.size(); i++) {
                bw.write("set all_cons[" + (i + 1) + "] := ");
                for (Event e : constraints.get(i).events) {
                    bw.write((e.index + 1) + " ");
                }
                bw.write(";\n");
            }
            bw.write("\n");

            ArrayList<Event> evets = new ArrayList<>();
            ArrayList<PrimtiveEvent> primits = new ArrayList<>();
            Set<Constraint> consts = new HashSet<>();
            for (Constraint c : constraints) {
                for (Event e : c.events) {
                    if (e.agent == agent) {
                        evets.add(e);
                        primits.addAll(e.pevents);
                        consts.add(c);
                    }
                }
            }

            bw.write("set agent_numcons := ");
            for (Constraint c : consts) {
                bw.write((c.index + 1) + " ");
            }
            bw.write(";\n");

            bw.write("set agent_numprims := ");
            for (PrimtiveEvent p : primits) {
                bw.write((p.index + 1) + " ");
            }
            bw.write(";\n");

            bw.write("set agent_numevents := ");
            for (Event e : evets) {
                bw.write((e.index + 1) + " ");
            }
            bw.write(";\n");

            bw.write("param agent_creward := ");
            for (Constraint c : consts) {
                bw.write((c.index + 1) + " " + (c.reward) + " ");
            }
            bw.write(";\n\n");

            bw.write("param agent_primitives : ");
            for (int i = 0; i < 4; i++) {
                bw.write((i + 1) + " ");
            }
            bw.write(":= \n");
            for (PrimtiveEvent p : primits) {
                bw.write((p.index + 1) + " " + (p.agent + 1) + " " + (p.state.index + 1) +
                        " " + (p.action.index + 1) + " " + (p.statedash.index + 1) + "\n");
            }
            bw.write(";\n");

            for (int i = 0; i < evets.size(); i++) {
                bw.write("set agent_events[" + (evets.get(i).index + 1) + "] := ");
                for (PrimtiveEvent p : evets.get(i).pevents) {
                    bw.write((p.index + 1) + " ");
                }
                bw.write(";\n");
            }
            bw.write("\n");

            for (Constraint c : consts) {
                bw.write("set agent_cons[" + (c.index + 1) + "] := ");
                for (Event e : c.events) {
                    if (e.agent == agent) {
                        bw.write((e.index + 1) + " ");
                    }
                }
                bw.write(";\n");
            }
            bw.write("\n\n");

            bw.write("param theta := " + (config.theta) + ";\n");
            bw.write("param Rmax := " + (config.Rmax) + ";\n");
            bw.write("param Rmin := " + (config.Rmin) + ";\n\n");
            bw.write("param initialxval := " + (config.initialxval) + ";\n");
            bw.write("for {(i,j,k,l) in sparseP} {	let P[i,j,k,l] := sparsePVal[i,j,k,l]; }");
            System.out.println("Done");
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (bw != null) {
                    bw.close();
                }
                if (fw != null) {
                    fw.close();
                }
            } catch (IOException e1) {
                e1.printStackTrace();
            }
        }
    }

    private void updateEMData(int agent, ArrayList<double[]> initx) {
        //System.out.print("      Updating AMPL for Agent " + (agent) + " : ");
        String FILENAME = Config.workDir + "Data/xdata" + (agent) + "_exp_" + (config.experiment) + ".dat";
        BufferedWriter bw = null;
        FileWriter fw = null;
        try {
            fw = new FileWriter(FILENAME);
            bw = new BufferedWriter(fw);

            bw.write("param x := \n");

            for (int k = 0; k < num_agents; k++) {
                bw.write("[" + (k + 1) + ",*,*] : ");
                for (Action a : mdps.get(k).actions) {
                    bw.write((a.index + 1) + " ");
                }
                bw.write(":= \n");
                for (State s : mdps.get(k).states) {
                    bw.write((s.index + 1) + " ");
                    for (Action a : mdps.get(k).actions) {
                        int ind = (s.index * mdps.get(k).numberActions) + a.index;
                        bw.write((initx.get(k)[(ind)]) + " ");
                    }
                    bw.write("\n");
                }
            }
            bw.write(";\n");

            //System.out.println("Done");
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (bw != null) {
                    bw.close();
                }
                if (fw != null) {
                    fw.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

    }

    private <T> T runWithTimeout(Callable<T> callable, long timeout, TimeUnit timeUnit) throws Exception {
        final ExecutorService executor = Executors.newSingleThreadExecutor();
        final Future<T> future = executor.submit(callable);
        executor.shutdown();
        try {
            return future.get(timeout, timeUnit);
        } catch (TimeoutException e) {
            future.cancel(true);
            throw e;
        } catch (ExecutionException e) {
            Throwable t = e.getCause();
            if (t instanceof Error) {
                throw (Error) t;
            } else if (t instanceof Exception) {
                throw e;
            } else {
                throw new IllegalStateException(t);
            }
        }
    }

    private ArrayList<double[]> runMultipleWithTimeout(ExecutorService executor, ArrayList<Callable<double[]>> taskList, Stopwatch ov_timer) throws InterruptedException, ExecutionException, TimeoutException {
        ArrayList<double[]> toret = new ArrayList<>();
        final List<Future<double[]>> future = executor.invokeAll(taskList);
        for (Future<double[]> f : future) {
            if (ov_timer.elapsed(TimeUnit.NANOSECONDS) > config.timeout*1e9) {
                throw new TimeoutException();
            }
            toret.add(f.get());
        }
        return toret;
    }

    private double objective(ArrayList<double[]> Rs, ArrayList<double[]> xvals) {
        double sum = 0;
        for (int i = 0; i < num_agents; i++) {
            double sumX = 0;
            double MDPsum = 0.0;
            assert Rs.get(i).length == xvals.get(i).length;
            for (int j = 0; j < Rs.get(i).length ; j++) {
                sumX += xvals.get(i)[j];
                MDPsum += Rs.get(i)[j] * xvals.get(i)[j];
            }
            sum += MDPsum;
            //System.out.println("XSUM: "+i+" "+sumX);
            //System.out.println("MDP Sum: " + i + " " + sum);
        }
        for (Constraint c : constraints) {
            double prod = c.reward;
            for (Event e : c.events) {
                double pesum = 0.0;
                int agent = e.agent;
                for (PrimtiveEvent p : e.pevents) {
                    State s = p.state;
                    Action a = p.action;
                    State sd = p.statedash;
                    int ind = (s.index * mdps.get(agent).numberActions) + a.index;
                    pesum += mdps.get(agent).transition(s, a, sd) * xvals.get(agent)[ind];
                }
                prod *= pesum;
            }
            sum += prod;
        }
        //System.out.println(sum);
        return sum;
    }

    private double[] firstEMIter(int agent) {
        try {
            AMPL ampl = ampls[agent];
            ampl.read(config.modelDir + "single.mod");
            ampl.readData(Config.workDir + "Data/single" + (agent) + "_exp_" + (config.experiment) + ".dat");
            ampl.setIntOption("solver_msg", 0);
            ampl.setOption("solver", "minos");
            ampl.solve();
            return ampl.getVariable("xstar").getValues().getColumnAsDoubles("val");
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    private double[] runSuccIter(int agent) {
        AMPL ampl = ampls[agent];
        updateEMData(agent, prev_x);
        ampl.eval("reset data x;");
        try {
            ampl.readData(Config.workDir + "Data/xdata" + (agent) + "_exp_" + (config.experiment) + ".dat");
        } catch (IOException e) {
            e.printStackTrace();
        }
        ampl.solve();
        return ampl.getVariable("xstar").getValues().getColumnAsDoubles("val");
    }

    private void EM(ArrayList<double[]> Rs, Double nonlinearobj, double nlptime) {
        int iter = 1;
        System.out.println("Iteration "+iter);

        double new_obj;
        double old_obj;
        Stopwatch ov_timer = Stopwatch.createUnstarted();
        Stopwatch iter_timer = Stopwatch.createUnstarted();

        final ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors() - 1);
        //AMPL[] ampls = new AMPL[num_agents];
        ArrayList<Callable<double[]>> taskList = new ArrayList<>();
        ArrayList<Double> change = new ArrayList<>();
        ArrayList<Double> results = new ArrayList<>();
        ArrayList<Double> times = new ArrayList<>();
        ArrayList<Double> newValues = new ArrayList<>();
        ArrayList<double[]> firstIter = new ArrayList<>();
        ov_timer.start();
        iter_timer.start();
        for (int i = 0; i < num_agents ; i++) {
            int var = i;
            ampls[i] = new AMPL();
            ampls[i].setOutputHandler(amplOutput -> {});
            taskList.add(() -> firstEMIter(var));
        }
        try {
            firstIter = runMultipleWithTimeout(executor, taskList, ov_timer);
            times.add((double)iter_timer.elapsed(TimeUnit.NANOSECONDS) / 1e9);
            if (firstIter != null) {
                new_obj = objective(Rs, firstIter);
                results.add(new_obj);
                System.out.println(ANSI_GREEN+"New Objective: " + new_obj+ANSI_RESET);
                System.out.println("\n"+ANSI_YELLOW+"Iteration "+iter+" time: "+iter_timer+ANSI_RESET);
                System.out.println(ANSI_YELLOW+"Overall EM Time: "+ov_timer+ANSI_RESET);
            } else {
                throw new Exception();
            }
            iter_timer.stop();

            for (int i = 0; i < num_agents ; i++) {
                ampls[i].setOption("solver", config.solver);
            }

            while (true) {
                iter += 1;

                if (iter % 50 == 0) {
                    Driver.saveToConsoleAndRedirect(ANSI_YELLOW+"Iteration : "+iter);
                    Driver.saveToConsoleAndRedirect(ANSI_GREEN+"Memory: "+getFreeSystemMemory());
                }

                System.out.println("Iteration "+iter);
                taskList.clear();

                iter_timer.reset();
                iter_timer.start();

                prev_x = firstIter;
                for (int i = 0; i < num_agents ; i++) {
                    int var = i;
                    taskList.add(() -> runSuccIter(var));
                }

                firstIter = runMultipleWithTimeout(executor, taskList, ov_timer);
                times.add((double)iter_timer.elapsed(TimeUnit.NANOSECONDS) / 1e9);
                if (firstIter != null) {
                    old_obj = new_obj;
                    new_obj = objective(Rs, firstIter);
                    results.add(new_obj);
                    iter_timer.stop();

                    System.out.println("\n"+ANSI_YELLOW+"Iteration "+iter+" time: "+iter_timer+ANSI_RESET);
                    System.out.println(ANSI_YELLOW+"Overall EM Time: "+ov_timer+ANSI_RESET);
                    System.out.println(ANSI_GREEN+"Old Objective: " + old_obj+ANSI_RESET);
                    System.out.println(ANSI_GREEN+"New Objective: " + new_obj+ANSI_RESET);
                    if (nonlinearobj != null) {
                        System.out.println(ANSI_CYAN+"Non Linear Objective: " + nonlinearobj+ANSI_RESET);
                        System.out.println(ANSI_PURPLE+"Percent Error: " +(Math.abs(new_obj-nonlinearobj)/nonlinearobj)*100+ANSI_RESET);
                    } else {
                        System.out.println(ANSI_CYAN+"Non Linear: NA"+ANSI_RESET);
                    }
                    change.add(Math.abs(new_obj-old_obj));
                    List<Double> tailChange = change.subList(Math.max(change.size() - config.noIterConvergenceCheck, 0), change.size());
                    Double sumChange = 0.0;
                    for(Double val : tailChange) {
                        sumChange += val;
                    }
                    Double avgChange = sumChange/config.noIterConvergenceCheck;

                    tailChange.clear();
                    newValues.clear();
                    System.gc();

                    System.out.println("Average Change: "+avgChange+"\n\n\n");
                    long time_spent = ov_timer.elapsed(TimeUnit.SECONDS);
                    if (avgChange <= config.deltaFinal || time_spent >= config.timeout) {
                        ov_timer.stop();
                        break;
                    }
                } else {
                    throw new Exception();
                }
            }
        } catch (Exception | OutOfMemoryError e) {
            System.err.println(e.getMessage());
            e.printStackTrace();

            if (e instanceof OutOfMemoryError) {
                System.out.println(ANSI_RED + "EM Ran Out of Memory." + ANSI_RESET);
                System.err.println("Max Memory: "+Driver.humanReadableByteCount(Runtime.getRuntime().maxMemory(), false));
                System.err.println("Free Memory: "+Driver.humanReadableByteCount(Runtime.getRuntime().freeMemory(), false));
                System.err.println("Total Memory: "+Driver.humanReadableByteCount(Runtime.getRuntime().totalMemory(), false));
                System.err.println("Used Memory: "+Driver.humanReadableByteCount((Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()), false));
            }

            if (e instanceof TimeoutException) {
                System.out.println(ANSI_RED + "EM Timed Out." + ANSI_RESET);
            }

            if (e instanceof AMPLException) {
                System.out.println(ANSI_RED + "AMPL ran into problem." + ANSI_RESET);
                System.err.println(((AMPLException) e).getLineNumber());
                System.err.println(e.getMessage());
                System.err.println(((AMPLException) e).getOffset());
                System.err.println(((AMPLException) e).getSourceName());
                System.err.println(e.toString());
            }

            if (e instanceof ExecutionException | e instanceof  InterruptedException) {
                System.out.println(ANSI_RED + "EM Ran into Exception :/" + ANSI_RESET);
            }

        } finally {
            mdps.clear();
            primitives.clear();
            events.clear();
            constraints.clear();
            for (int i = 0; i < num_agents ; i++) {
                ampls[i].close();
            }
            executor.shutdownNow();
            System.out.println(ANSI_YELLOW+"Overall EM Time: "+ov_timer+ANSI_RESET);
            System.out.println(ANSI_YELLOW+"Overall NLP Time: "+nlptime+" sec"+ANSI_RESET);
            double ovEMTime = ov_timer.elapsed(TimeUnit.NANOSECONDS) / 1e9;
            if (results.size() >= 1 && nonlinearobj != null) {
                XYLineChart.genAndSaveGraph(config.experiment + "", results, nonlinearobj, times, nlptime, ovEMTime);
            } else if (results.size() >= 1) {
                XYLineChart.genAndSaveGraph(config.experiment + "", results, 0.0, times, nlptime, ovEMTime);
            } else {
                System.out.println("Nothing to plot.");
            }
            ov_timer.reset();
            iter_timer.reset();
            change.clear();
            results.clear();
            times.clear();
            firstIter.clear();
            newValues.clear();
            prev_x = null;
            System.gc();
        }
    }

    void runExperiment(boolean nlp) {

        Stopwatch timer = Stopwatch.createUnstarted();
        ArrayList<double[]> Rs = new ArrayList<>();
        Double nonlinearobj = null;
        for (int i = 0; i < num_agents; i++) {
            ArrayList<double[]> ret = mdps.get(i).generateLPAc();
            Rs.add(ret.get(0));
        }

        if (config.GenRun == 1) {
            genAMPL();
            for (int i = 0; i < num_agents; i++) {
                genAMPLSingle(i);
            }
        }

        Driver.saveToConsoleAndRedirect("NLP Start: "+new Date());

        System.out.println("----+ Non Linear +---- ");
        timer.start();
        try {
            System.out.print(ANSI_YELLOW);
            if (nlp) {
                nonlinearobj = runWithTimeout(this::NonLinear, config.timeout, TimeUnit.SECONDS);
            }
            if (nonlinearobj != null) {
                System.out.println("\nNon Linear Objective Value: " + nonlinearobj + ANSI_RESET);
            }
        } catch (Exception e) {
            try {
                // FIXME: 7/6/17 Test out on other platforms and can try other elegant solutions.
                Process p = Runtime.getRuntime().exec(new String[]{"sh", "-c", "ps aux | grep libs/../ampl/ampl | awk 'NR==1{ print $2; }'"});
                //Process p1 = Runtime.getRuntime().exec(new String[]{"sh", "-c", "pkill -f ampl"});
                //Process p2 = Runtime.getRuntime().exec(new String[]{"sh", "-c", "pkill -f bonmin"});
                BufferedReader stdInput = new BufferedReader(new InputStreamReader(p.getInputStream()));
                String s = stdInput.readLine();
                Runtime.getRuntime().exec(new String[]{"sh", "-c", "kill -15 "+s});
            } catch (IOException e1) {
                e1.printStackTrace();
            }
            System.out.println(ANSI_RED + "Non Linear Timed Out / Ran Out of Memory." + ANSI_RESET);
            //e.printStackTrace();
        }
        timer.stop();
        double nlptime = timer.elapsed(TimeUnit.NANOSECONDS);
        System.out.println(ANSI_GREEN + "Non Linear Took: " + timer + ANSI_RESET);
        Driver.saveToConsoleAndRedirect("NLP Done And EM starts: "+new Date());

        System.out.println("\n-------+ EM +-------\n");
        EM(Rs, nonlinearobj, nlptime/1e9);
        Driver.saveToConsoleAndRedirect("EM Done: "+new Date());
    }

    private void runConfigNonLinear() {
        System.out.print("     Writing running config for NLP: ");
        String FILENAME = Config.workDir + "Data/nl2_exp_" + config.experiment + ".run";
        BufferedWriter bw = null;
        FileWriter fw = null;
        try {
            fw = new FileWriter(FILENAME);
            bw = new BufferedWriter(fw);
            bw.write("reset;\n");
            bw.write("model "+Config.modelDir+"try.mod;\n");
            bw.write("data " + Config.workDir+ "Data/nl2_exp_"+ config.experiment + ".dat" + ";\n");
            bw.write("option solver_msg 0;\n");
            bw.write("write \"g/"+Config.workDir+"Data/myfile"+ config.experiment +"\";\n");
            System.out.println("Done");
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (bw != null) {
                    bw.close();
                }
                if (fw != null) {
                    fw.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private void updateRunConfigNonLinear() {
        String FILENAME = Config.workDir + "Data/nl2_exp_" + config.experiment + ".run";
        BufferedWriter bw = null;
        FileWriter fw = null;
        try {
            fw = new FileWriter(FILENAME);
            bw = new BufferedWriter(fw);
            bw.write("reset;\n");
            bw.write("model "+Config.modelDir+"try.mod;\n");
            bw.write("data " + Config.workDir+ "Data/nl2_exp_"+ config.experiment + ".dat" + ";\n");
            bw.write("option solver_msg 0;\n");
            bw.write("solution \""+Config.workDir+"Data/myfile"+ config.experiment +".sol\";\n");
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (bw != null) {
                    bw.close();
                }
                if (fw != null) {
                    fw.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private Double NonLinear() {
//        ArrayList<double[]> initx = new ArrayList<>();
//        ArrayList<double[]> Rs = new ArrayList<>();
//        for (int i = 0; i < num_agents; i++) {
//            ArrayList<double[]> ret = mdps.get(i).generateLPAc();
//            Rs.add(ret.get(0));
//        }
        AMPL ampl = new AMPL();
        ampl.reset();
        ampl.setOutputHandler(amplOutput -> {});
        String FILENAME = Config.workDir + "Data/nl2_exp_" + config.experiment + ".run";
        Double retval = null;
        try {
            Process p1 = Runtime.getRuntime().exec(new String[]{"sh", "-c", "rm -f "+Config.workDir+"Data/myfile"+config.experiment+".nl"});
            Process p2 = Runtime.getRuntime().exec(new String[]{"sh", "-c", "rm -f "+Config.workDir+"Data/myfile"+config.experiment+".sol"});
            runConfigNonLinear();
            ampl.read(FILENAME);
            Process p3 = Runtime.getRuntime().exec(new String[]{"sh", "-c", "./ampl/bonmin -s "+Config.workDir+"Data/myfile"+config.experiment+".nl"});
            ampl.reset();
            updateRunConfigNonLinear();
            ampl.read(FILENAME);
            retval = ampl.getObjective("ER").value();
//            Variable var = ampl.getVariable("x");
//            DataFrame var_vals = var.getValues();
//            int over_ind = 0;
//            for (int i = 0; i < num_agents ; i++) {
//                int ind = 0;
//                double arr[] = new double[mdps.get(i).numberVariables];
//                for (int j = 0; j < arr.length; j++) {
//                    Object[] retarr = var_vals.getRowByIndex(over_ind);
//                    over_ind += 1;
//                    arr[ind] = (Double) retarr[retarr.length - 1];
//                    ind += 1;
//                }
//                initx.add(arr);
//            }
//            objective(Rs,initx);
            //System.out.println(obj.value());
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            ampl.close();
            System.gc();
        }
        return retval;
    }

    public Double getRandomPolicyValue() {
        ArrayList<double[]> initx = new ArrayList<>();
        ArrayList<double[]> Rs = new ArrayList<>();
        for (int i = 0; i < num_agents; i++) {
            ArrayList<double[]> ret = mdps.get(i).generateLPAc();
            Rs.add(ret.get(0));
        }
        AMPL ampl = new AMPL();
        ampl.reset();
        Double retval = null;

        try {
            ampl.read(Config.modelDir+"try_random.mod");
            ampl.readData(Config.workDir+"Data/nl2_exp_"+ config.experiment + ".dat");
            ampl.setOption("solver", "bonmin");
            ampl.solve();
            Variable var = ampl.getVariable("x");
            DataFrame var_vals = var.getValues();
            int over_ind = 0;
            for (int i = 0; i < num_agents ; i++) {
                int ind = 0;
                double arr[] = new double[mdps.get(i).numberVariables];
                for (int j = 0; j < arr.length; j++) {
                    Object[] retarr = var_vals.getRowByIndex(over_ind);
                    over_ind += 1;
                    arr[ind] = (Double) retarr[retarr.length - 1];
                    ind += 1;
                }
                initx.add(arr);
            }
            retval = objective(Rs, initx);
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            ampl.close();
            System.gc();
        }
        return retval;
    }


}
