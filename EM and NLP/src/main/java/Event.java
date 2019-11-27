import java.util.ArrayList;

/**
 * Created by tarun on 5/6/17.
 */
public class Event {
    int agent;
    ArrayList<PrimtiveEvent> pevents;
    int index;
    String name;
    int site;

    public Event(int agent, ArrayList<PrimtiveEvent> pevents, int index, String name, int site) {
        this.agent = agent;
        this.pevents = pevents;
        this.index = index;
        this.name = name;
        this.site = site;
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof Event)) {
            return false;
        }
        Event other = (Event) o;

        return this.index==other.index &&
                this.agent==other.agent &&
                this.name.equals(other.name)&&
                this.site==other.site;
    }

    @Override
    public String toString() {
        return "E: ( " + agent + " " + pevents.toString() + " " + name + " " + site + " )";
    }
}

