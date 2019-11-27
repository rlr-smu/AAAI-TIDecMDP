import java.util.ArrayList;

/**
 * Created by tarun on 5/6/17.
 */
public class Constraint {
    ArrayList<Event> events;
    double reward;
    int index;

    public Constraint(ArrayList<Event> events, double rew, int ind) {
        this.events = events;
        this.reward = rew;
        this.index = ind;
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof Constraint)) {
            return false;
        }
        Constraint other = (Constraint) o;

        return this.index==other.index &&
                this.reward==other.reward;
    }
}
