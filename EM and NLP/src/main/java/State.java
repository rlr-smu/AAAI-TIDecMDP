/**
 * Created by tarun on 4/6/17.
 */

import java.io.Serializable;
import java.util.ArrayList;

public class State implements Serializable {
    int index;
    int location;
    int actualLocation;
    int time;
    String dvals;
    int dold;
    ArrayList<Action> possibleActions;
    ArrayList<Transition> transition;
    ArrayList<Reward> reward;

    public State(int ind, int location, int actLocation, int time, String dvals, int dold, ArrayList<Action> actions) {
        this.index = ind;
        this.location = location;
        this.actualLocation = actLocation;
        this.time = time;
        this.dvals = dvals;
        this.dold = dold;
        this.possibleActions = actions;
        this.transition = new ArrayList<>();
        this.reward = new ArrayList<>();
    }

    public void setTransition(ArrayList<Transition> tr) {
        this.transition = tr;
    }

    public ArrayList<Transition> getTransition() {
        return this.transition;
    }

    public void setReward(ArrayList<Reward> rew) {
        this.reward = rew;
    }

    public ArrayList<Reward> getReward() {
        return this.reward;
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof State)) {
            return false;
        }
        State other = (State) o;

        return this.index==other.index &&
                this.location==other.location &&
                this.actualLocation==other.actualLocation &&
                this.time==other.time &&
                this.dold==other.dold &&
                this.dvals.equals(other.dvals);
    }

    @Override
    public String toString() {
        return "Index: " + this.index + " Location: " + this.location + " Actual: " + this.actualLocation + " Time: " + this.time + " Dvals " + this.dvals + " Dold: " + this.dold;
    }

}
