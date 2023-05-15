import mpi.MPI;
import mpi.Status;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.List;
import java.util.Stack;
import java.util.concurrent.TimeUnit;

public class ConnectFour {

  public static final char COMPUTER = 'C';
  public static final char HUMAN = 'H';
  public static final int MASTER = 0;
  public static final int HEIGHT = 6;
  public static final int WIDTH = 7;
  public static final int MAX_DEPTH = 7;
  public static final char[] REQUEST_STATE = "REQUEST_STATE".toCharArray();
  public static final int TAG_REQUEST = 0;
  public static final int TAG_RESPONSE = 1;
  public static final int portNumber = 6868;

  public static final int NUMBER_OF_TASKS = 7 * 7;


  public String bestMove = "";
  public float highScore = -1;
  public int count;
  private final Runnable responseHandler;


  public static void main(String[] args) {
    ConnectFour connectFour = new ConnectFour();

    connectFour.play(args);
  }

  public ConnectFour() {
    this.responseHandler = () -> {
      while (count < NUMBER_OF_TASKS) {
        Status status = MPI.COMM_WORLD.Probe(MPI.ANY_SOURCE, TAG_RESPONSE);
        byte[] workResultChars = new byte[status.count];

        MPI.COMM_WORLD.Recv(workResultChars, 0, workResultChars.length, MPI.BYTE, status.source, status.tag);

        WorkResult work = Util.deserialize(workResultChars, WorkResult.class);

        if (work.score >= highScore) {
          highScore = work.score;
          bestMove = work.state.substring(work.state.length() - 2, work.state.length() - 1);
        }

        count++;
      }

      synchronized (this) {
        notify();
      }
    };
  }

  public void play(String[] args) {
    MPI.Init(args);

    int currentIndex = MPI.COMM_WORLD.Rank();
//    int numberOfProcesses = MPI.COMM_WORLD.Size();


    if (currentIndex == MASTER) {
      StringBuilder moves = new StringBuilder();

      System.out.println("Attempting to make server socket");

      try (
              ServerSocket serverSocket = new ServerSocket(portNumber, 50, InetAddress.getByName("0.0.0.0"));
              Socket clientSocket = serverSocket.accept();
              BufferedReader in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
      ) {
        System.out.println("Created socket " + serverSocket);
        while (true) {
          System.out.println("Which field do you choose?");

          String inputLine = in.readLine();

          System.out.printf("Human is playing column: %s\n", inputLine);
          int move = Integer.parseInt(inputLine.trim());

          if (move < 1 || move > 7) {
            System.out.println("Illegal move");
            MPI.Finalize();
            return;
          }

          moves.append(move);

          Stack<String> states = new Stack<>();

          // Generating 49 subtrees representing tasks for the slaves
          for (int i = 1; i <= 7; i++) {
            for (int j = 1; j <= 7; j++) {
              String state = moves + Util.convertListToString(List.of(i, j));
              states.push(state);
            }
          }

          highScore = 0;
          bestMove = "";
          count = 0;

          Thread responseHandlerThread = new Thread(responseHandler);
          responseHandlerThread.start();

          long startTime = System.nanoTime();

          while (!states.isEmpty()) {
            // When a slave requests work we provide
            Status status = MPI.COMM_WORLD.Probe(MPI.ANY_SOURCE, TAG_REQUEST);
            char[] messageChars = new char[status.count];

            MPI.COMM_WORLD.Recv(messageChars, 0, messageChars.length, MPI.CHAR, status.source, status.tag);

            String state = states.pop();

//            System.out.printf("Master (%d) responding to work request #%d with state %s\n", currentIndex, count, state);

            char[] stateCharArray = state.toCharArray();

            MPI.COMM_WORLD.Send(stateCharArray, 0, stateCharArray.length, MPI.CHAR, status.source, TAG_REQUEST);
          }


          do {
            try {
              synchronized (this) {
                // Barrier - waiting for all tasks to be finished (all subtrees to be evaluated)
                this.wait(10000);
              }
            } catch (InterruptedException ignored) {
            }
          } while (count < NUMBER_OF_TASKS);

          long endTime = System.nanoTime();
          long duration = endTime - startTime;
          long durationInMilliseconds = TimeUnit.NANOSECONDS.toMillis(duration);

          System.out.println("Execution time: " + durationInMilliseconds + " ms");

          if (bestMove.isEmpty()) {
            // He is indifferent at this point.
            bestMove = "1";
          }

          moves.append(bestMove);

          System.out.println("Best score: " + highScore);
          System.out.println("Computers move: " + bestMove);

          responseHandlerThread.interrupt();

          char[][] board = Util.convertMovesToBoard(moves.toString());
          Util.printBoard(board);

          char winner = Util.checkWinner(board);

          if (winner == HUMAN) {
            System.out.println("Human won!");
            break;
          } else if (winner == COMPUTER) {
            System.out.println("Computer won!");
            break;
          }
        }
      } catch (IOException e) {
        System.out.println("Exception caught when trying to listen on port " + portNumber);
        System.out.println(e.getMessage());
      }
    } else {
      while (true) {
//        System.out.printf("Worker %d requesting work\n", currentIndex);

        // SLAVE
        MPI.COMM_WORLD.Send(REQUEST_STATE, 0, REQUEST_STATE.length, MPI.CHAR, MASTER, TAG_REQUEST);


        // The Master responds with a state for us to process
        Status status = MPI.COMM_WORLD.Probe(MPI.ANY_SOURCE, TAG_REQUEST);
        char[] messageChars = new char[status.count];

        MPI.COMM_WORLD.Recv(messageChars, 0, messageChars.length, MPI.CHAR, status.source, status.tag);
        String currentWorkloadState = String.valueOf(messageChars);

//        System.out.printf("Worker %d received state: %s\n", currentIndex, currentWorkloadState);

        float score = evaluateState(new GameState(currentWorkloadState), 0);

//        System.out.printf("Worker %d evaluated score: %f\n", currentIndex, score);

        WorkResult result = new WorkResult(score, currentWorkloadState);

        byte[] resultBytes = Util.serialize(result);

        MPI.COMM_WORLD.Send(resultBytes, 0, resultBytes.length, MPI.BYTE, MASTER, TAG_RESPONSE);
      }
    }

    MPI.Finalize();
  }

  public float evaluateState(GameState state, int depth) {
    if (depth == MAX_DEPTH) {
      return state.score;
    }

    // Iterate over all possible moves
    for (int i = 1; i <= 7; i++) {
      String nextMoves = state.moves + i;
      char[][] board = Util.convertMovesToBoard(nextMoves);

      // if the move is invalid skip it
      if (board[0][i - 1] != ' ') {
        continue;
      }

      char winner = Util.checkWinner(board);

      if (winner == COMPUTER && nextMoves.length() % 2 == 1) {
        state.addScore(1);
      } else if (winner == HUMAN && nextMoves.length() % 2 == 0) {
        state.addScore(-1);
      } else {
        GameState childState = new GameState(nextMoves);
        state.addScore(evaluateState(childState, depth + 1));
      }
    }

    return state.getAverageScore();
  }


}
