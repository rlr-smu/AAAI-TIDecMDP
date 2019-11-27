/**
 * Created by tarun on 5/6/17.
 */
public class PrimtiveEvent {
    int agent;
    State state;
    Action action;
    State statedash;
    int index;

    public PrimtiveEvent(int agent, State s, Action a, State sd, int index) {
        this.agent = agent;
        this.state = s;
        this.action = a;
        this.statedash = sd;
        this.index = index;
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof PrimtiveEvent)) {
            return false;
        }
        PrimtiveEvent other = (PrimtiveEvent) o;

        return this.index==other.index &&
                this.agent==other.agent &&
                this.state.equals(other.state)&&
                this.action.equals(other.action)&&
                this.statedash.equals(other.statedash);
    }

    @Override
    public String toString() {
        return "PE: Agent: " + agent + " Index: " + index + " State: " + state + " Action: " + action + " Statedash: " + statedash;
    }
}
