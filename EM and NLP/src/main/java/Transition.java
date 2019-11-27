import java.io.Serializable;

/**
 * Created by tarun on 5/6/17.
 */
public class Transition implements Serializable {
    int action_index;
    int statedash_index;
    double probability;

    public Transition(int action_index, int statedash_index, double probability) {
        this.action_index = action_index;
        this.statedash_index = statedash_index;
        this.probability = probability;
    }
}