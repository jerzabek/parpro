import java.io.*;
import java.util.List;


public class Util {

  /**
   * Utility method to serialize object sent over MPJ as a byte array.
   */
  public static byte[] serialize(Serializable s) {
    try (ByteArrayOutputStream bos = new ByteArrayOutputStream(); ObjectOutput out = new ObjectOutputStream(bos)) {
      out.writeObject(s);
      out.flush();
      return bos.toByteArray();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * Utility method to deserialize object received over MPJ as a byte array.
   */
  @SuppressWarnings("unchecked")
  public static <T extends Serializable> T deserialize(byte[] b, Class<T> cl) {
    try (ByteArrayInputStream bis = new ByteArrayInputStream(b)) {
      ObjectInput in;
      in = new ObjectInputStream(bis);
      return (T) in.readObject();
    } catch (IOException | ClassNotFoundException e) {
      throw new RuntimeException(e);
    }
  }


  public static char checkWinner(char[][] board) {
    char[] players = {ConnectFour.HUMAN, ConnectFour.COMPUTER};

    for (char player : players) {
      // check horizontal lines
      for (int j = 0; j < ConnectFour.HEIGHT; j++) {
        for (int i = 0; i < ConnectFour.WIDTH - 3; i++) {
          if (board[j][i] == player && board[j][i + 1] == player && board[j][i + 2] == player && board[j][i + 3] == player)
            return player;
        }
      }

      // check vertical lines
      for (int i = 0; i < ConnectFour.WIDTH; i++) {
        for (int j = 0; j < ConnectFour.HEIGHT - 3; j++) {
          if (board[j][i] == player && board[j + 1][i] == player && board[j + 2][i] == player && board[j + 3][i] == player)
            return player;
        }
      }

      // check diagonals from left to right
      for (int i = 3; i < ConnectFour.WIDTH; i++) {
        for (int j = 0; j < ConnectFour.HEIGHT - 3; j++) {
          if (board[j][i] == player && board[j + 1][i - 1] == player && board[j + 2][i - 2] == player && board[j + 3][i - 3] == player)
            return player;
        }
      }

      // check diagonals from right to left
      for (int i = 3; i < ConnectFour.WIDTH; i++) {
        for (int j = 3; j < ConnectFour.HEIGHT; j++) {
          if (board[j][i] == player && board[j - 1][i - 1] == player && board[j - 2][i - 2] == player && board[j - 3][i - 3] == player)
            return player;
        }
      }
    }

    // If no one has won yet, return ' '
    return ' ';
  }


  public static char[][] convertMovesToBoard(String moves) {
    char[][] board = new char[ConnectFour.HEIGHT][ConnectFour.WIDTH];

    // Initialize the board with ' ' to represent empty cells
    for (int i = 0; i < ConnectFour.HEIGHT; i++) {
      for (int j = 0; j < ConnectFour.WIDTH; j++) {
        board[i][j] = ' ';
      }
    }

    // Iterate over the moves
    for (int k = 0; k < moves.length(); k++) {
      int column = Character.getNumericValue(moves.charAt(k)) - 1;
      char player = (k % 2 == 0) ? ConnectFour.HUMAN : ConnectFour.COMPUTER;

      // Find the lowest empty cell in the chosen column
      for (int i = ConnectFour.HEIGHT - 1; i >= 0; i--) {
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
    for (int i = 0; i < ConnectFour.HEIGHT; i++) {
      for (int j = 0; j < ConnectFour.WIDTH; j++) {
        System.out.print("| " + board[i][j] + " ");
      }
      System.out.println("|");
    }

    // Print bottom line
    for (int j = 0; j < ConnectFour.WIDTH; j++) {
      System.out.print("----");
    }

    System.out.println("-");

    // Print bottom line
    for (int j = 0; j < ConnectFour.WIDTH; j++) {
      System.out.printf("  %d ", j + 1);
    }

    System.out.println();
  }

}
