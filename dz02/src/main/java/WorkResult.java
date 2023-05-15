import java.io.Serializable;

public class WorkResult implements Serializable {

  public float score;
  public String state;

  public WorkResult(float score, String state) {
    this.score = score;
    this.state = state;
  }
}