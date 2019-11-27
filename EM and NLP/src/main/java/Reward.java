import java.io.Serializable;

/**
 * Created by tarun on 5/6/17.
 */
public class Reward implements Serializable {
    int action_index;
    double reward;

    public Reward(int action_index, double prob) {
        this.action_index = action_index;
        this.reward = prob;
    }
}
