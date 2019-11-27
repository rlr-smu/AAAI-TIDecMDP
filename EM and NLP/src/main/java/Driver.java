import java.io.*;
import java.util.Date;

/**
 * Created by tarun on 11/6/17.
 */

public class Driver {

    private static final String ANSI_BLACK_BACKGROUND = "\u001B[40m";
    private static final String ANSI_RED_BACKGROUND = "\u001B[41m";
    private static final String ANSI_GREEN_BACKGROUND = "\u001B[42m";
    private static final String ANSI_YELLOW_BACKGROUND = "\u001B[43m";
    private static final String ANSI_BLUE_BACKGROUND = "\u001B[44m";
    private static final String ANSI_PURPLE_BACKGROUND = "\u001B[45m";
    private static final String ANSI_CYAN_BACKGROUND = "\u001B[46m";
    private static final String ANSI_WHITE_BACKGROUND = "\u001B[47m";
    private static final String ANSI_RESET = "\u001B[0m";
    private static PrintStream logger_;

    public static void main( String[ ] args ) {
        String line = "";
        try (BufferedReader reader = new BufferedReader(new FileReader(Config.workDir+"experiments/Domain1Easy.csv"))) {
            while ((line = reader.readLine()) != null ) {
                String[] experimentSettings = line.split(",");
                assert experimentSettings.length == 7;
                int experiment = Integer.parseInt(experimentSettings[0]);

                saveToConsoleAndRedirect("Max Memory: "+humanReadableByteCount(Runtime.getRuntime().maxMemory(), false));
                saveToConsoleAndRedirect("Free Memory: "+humanReadableByteCount(Runtime.getRuntime().freeMemory(), false));
                saveToConsoleAndRedirect("Total Memory: "+humanReadableByteCount(Runtime.getRuntime().totalMemory(), false));
                saveToConsoleAndRedirect("Used Memory: "+humanReadableByteCount((Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()), false));
                saveToConsoleAndRedirect("");

                logger_ = new PrintStream(new BufferedOutputStream(new FileOutputStream(Config.workDir+"Logs/output"+experiment+".log")), true);
                System.setErr(new PrintStream(new BufferedOutputStream(new FileOutputStream(Config.workDir+"Logs/outputErr"+experiment+".log")), true));

                saveToConsoleAndRedirect(ANSI_BLUE_BACKGROUND+"Starting Experiment "+experiment+": ");

                int agents = Integer.parseInt(experimentSettings[1]);
                int nPrivatePerAgent = Integer.parseInt(experimentSettings[2]);
                int nShared = Integer.parseInt(experimentSettings[3]);
                int minSharing = Integer.parseInt(experimentSettings[4]);
                int maxSharing = minSharing;
                int minT = Integer.parseInt(experimentSettings[5]);
                int maxT = minT;
                int minTaction = Integer.parseInt(experimentSettings[6]);
                int maxTaction = minTaction;
                //Config c = new Config(experiment, agents, nPrivatePerAgent, nShared, minSharing, maxSharing, minT, maxT, minTaction, maxTaction);
                Config c = new Config(experiment);

                saveToConsoleAndRedirect(ANSI_RED_BACKGROUND+"Generation Start: "+new Date());

                EDECMDP edecmdp = new EDECMDP(c);

                System.gc();

                saveToConsoleAndRedirect(ANSI_GREEN_BACKGROUND+"Generation Done: "+new Date());

                saveToConsoleAndRedirect(edecmdp.number_of_variables+"");

                saveToConsoleAndRedirect(""+edecmdp.getRandomPolicyValue());
                //edecmdp.runExperiment(true);

                logger_.close();

                System.gc();
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    static void saveToConsoleAndRedirect(String outs) {
        System.setOut(new PrintStream(new FileOutputStream(FileDescriptor.out)));
        System.out.println("\t" + outs + ANSI_RESET);
        System.setOut(logger_);
    }

    static String humanReadableByteCount(long bytes, boolean si) {
        int unit = si ? 1000 : 1024;
        if (bytes < unit) return bytes + " B";
        int exp = (int) (Math.log(bytes) / Math.log(unit));
        String pre = (si ? "kMGTPE" : "KMGTPE").charAt(exp-1) + (si ? "" : "i");
        return String.format("%.1f %sB", bytes / Math.pow(unit, exp), pre);
    }
}
