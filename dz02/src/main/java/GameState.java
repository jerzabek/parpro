public class GameState {
  String moves;
  float score;
  int numChildren;

  public GameState(String moves) {
    this.moves = moves;
    this.score = 0;
    this.numChildren = 0;
  }

  public void addScore(float score) {
    this.score += score;
    this.numChildren++;
  }

  public float getAverageScore() {
    return numChildren == 0 ? 0 : score / numChildren;
  }
}
