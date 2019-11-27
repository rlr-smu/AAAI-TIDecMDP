import java.io.Serializable;

/**
 * Created by tarun on 4/6/17.
 */
public class Action implements Serializable {
    int index;
    int gotox;
    String name;

    public Action(int ind, String name) {
        this(ind, name, -1);
    }

    public Action(int ind, String name, int gotox) {
        this.index = ind;
        this.name = name;
        this.gotox = gotox;
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof Action)) {
            return false;
        }
        Action other = (Action)o;

        return (other.index==this.index && other.name.equals(this.name) && other.gotox==this.gotox);
    }

    @Override
    public String toString() {
        return "Index: " + this.index + " Name: " + this.name + " Goto: " + this.gotox;
    }
}
