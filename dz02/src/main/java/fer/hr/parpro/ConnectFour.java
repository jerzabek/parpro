package fer.hr.parpro;

import java.util.List;
import java.util.Scanner;

public class ConnectFour {
  public static final int MASTER = 0;

  public static final int HEIGHT = 6;
  public static final int WIDTH = 7;

  public static final int MAX_DEPTH = 7;


  private static int currentIndex = MASTER, numberOfPhilosophers;

  static class GameState {
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

  public static void main(String[] args) {
//    MPI.Init(args);
//
//    currentIndex = MPI.COMM_WORLD.Rank();
//    numberOfPhilosophers = MPI.COMM_WORLD.Size();


    if (currentIndex == MASTER) {
      Scanner s = new Scanner(System.in);

      StringBuilder moves = new StringBuilder();

      while (true) {
        System.out.println("Which field do you choose?");
        int move = s.nextInt(); // User makes the first move

        if (move < 1 || move > 7) {
          System.out.println("Illegal move");
          // TODO: MPI die
          return;
        }

        moves.append(move);

        // Generate 49 tasks

//      Stack<String> states = new Stack<>();


        float highScore = 0;
        String bestMove = "";

        for (int i = 1; i <= 7; i++) {
          for (int j = 1; j <= 7; j++) {
            String state = moves + convertListToString(List.of(i, j));

            float score = evaluateState(new GameState(state), 0);

            if (score >= highScore) {
              highScore = score;
              bestMove = String.valueOf(i);
            }
          }
        }

        if (bestMove.isEmpty()) {
          // He is indifferent at this point.
          bestMove = "1";
        }

        moves.append(bestMove);

        System.out.println("Computers move: " + bestMove);

        char[][] board = convertMovesToBoard(moves.toString());
        printBoard(board);

        char winner = checkWinner(board);

        if (winner == 'H') {
          System.out.println("Human won!");
          break;
        } else if (winner == 'C') {
          System.out.println("Computer won!");
          break;
        }
      }

      // Wait for work requests
      // Hand out state
      // A slave solves the given state, returning a number?

    }
  }

  public static float evaluateState(GameState state, int depth) {
    if (depth == MAX_DEPTH) {
      return state.score;
    }

    // Iterate over all possible moves (7 columns)
    for (int i = 1; i <= 7; i++) {
      String nextMoves = state.moves + i;
      char[][] board = convertMovesToBoard(nextMoves);

      // Check if the move is valid (column is not full)
      // This whole thing just equals i
      int column = Integer.parseInt(nextMoves.substring(nextMoves.length() - 1));
      if (board[0][column - 1] != ' ') {
        continue;  // Skip this iteration if the move is not valid
      }

      char winner = checkWinner(board);

      if (winner == 'C' && nextMoves.length() % 2 == 1) {
        state.addScore(1);
      } else if (winner == 'H' && nextMoves.length() % 2 == 0) {
        state.addScore(-1);
      } else {
        GameState childState = new ConnectFour.GameState(nextMoves);
        state.addScore(evaluateState(childState, depth + 1));
      }
    }

    return state.getAverageScore();
  }

  public static char checkWinner(char[][] board) {
    int HEIGHT = board.length;
    int WIDTH = board[0].length;
    char[] players = {'H', 'C'};

    for (char player : players) {
      // check horizontal lines
      for (int j = 0; j < HEIGHT; j++) {
        for (int i = 0; i < WIDTH - 3; i++) {
          if (board[j][i] == player && board[j][i + 1] == player && board[j][i + 2] == player && board[j][i + 3] == player)
            return player;
        }
      }

      // check vertical lines
      for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < HEIGHT - 3; j++) {
          if (board[j][i] == player && board[j + 1][i] == player && board[j + 2][i] == player && board[j + 3][i] == player)
            return player;
        }
      }

      // check diagonals from left to right
      for (int i = 3; i < WIDTH; i++) {
        for (int j = 0; j < HEIGHT - 3; j++) {
          if (board[j][i] == player && board[j + 1][i - 1] == player && board[j + 2][i - 2] == player && board[j + 3][i - 3] == player)
            return player;
        }
      }

      // check diagonals from right to left
      for (int i = 3; i < WIDTH; i++) {
        for (int j = 3; j < HEIGHT; j++) {
          if (board[j][i] == player && board[j - 1][i - 1] == player && board[j - 2][i - 2] == player && board[j - 3][i - 3] == player)
            return player;
        }
      }
    }

    // If no one has won yet, return ' '
    return ' ';
  }


  public static char[][] convertMovesToBoard(String moves) {
    char[][] board = new char[HEIGHT][WIDTH];

    // Initialize the board with ' ' to represent empty cells
    for (int i = 0; i < HEIGHT; i++) {
      for (int j = 0; j < WIDTH; j++) {
        board[i][j] = ' ';
      }
    }

    // Iterate over the moves
    for (int k = 0; k < moves.length(); k++) {
      int column = Character.getNumericValue(moves.charAt(k)) - 1;  // Convert to 0-based index
      char player = (k % 2 == 0) ? 'H' : 'C';  // Alternate between 'H' and 'C'

      // Find the lowest empty cell in the chosen column
      for (int i = HEIGHT - 1; i >= 0; i--) {
        if (board[i][column] == ' ') {
          board[i][column] = player;
          break;
        }
      }
    }

    return board;
  }

  public static String convertListToString(List<Integer> list) {
    StringBuilder sb = new StringBuilder();
    for (Integer num : list) {
      sb.append(num);
    }
    return sb.toString();
  }

  public static void printBoard(char[][] board) {
    for (int i = 0; i < HEIGHT; i++) {
      for (int j = 0; j < WIDTH; j++) {
        System.out.print("| " + board[i][j] + " ");
      }
      System.out.println("|");
    }

    // Print bottom line
    for (int j = 0; j < WIDTH; j++) {
      System.out.print("----");
    }

    System.out.println("-");

    // Print bottom line
    for (int j = 0; j < WIDTH; j++) {
      System.out.printf("  %d ", j + 1);
    }

    System.out.println();
  }


}
