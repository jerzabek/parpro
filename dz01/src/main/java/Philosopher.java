import mpi.MPI;
import mpi.Status;

import java.io.*;

public abstract class Philosopher {

  protected final static int MIN_THINK_DURATION = 2000;
  protected final static int MAX_THINK_DURATION = 5000;

  protected boolean[] forkRequests = new boolean[2];

  public abstract void think();

  public abstract void acquireForks();

  public abstract void eat();

  /**
   * By default, all philosophers have dirty forks.
   */
  protected enum Fork implements Serializable {
    CLEAN, DIRTY, REQUEST
  }

  public Fork leftFork = null;
  public Fork rightFork = null;

  protected int currentIndex, numberOfPhilosophers;

  public Philosopher(String[] args) {
    initializePhilosopher(args);

    Thread checkForkRequests = new Thread(() -> {
      while (true) {
        log("Waiting for messages...");
        Status status = MPI.COMM_WORLD.Probe(MPI.ANY_SOURCE, MPI.ANY_TAG);

        log("Received message...");

        if (status != null) {
          log("Received message from %d", status.source);

          byte[] receivedBytes = new byte[status.count];
          MPI.COMM_WORLD.Recv(receivedBytes, 0, status.count, MPI.BYTE, MPI.ANY_SOURCE, MPI.ANY_TAG);

          Fork receivedFork = deserialize(receivedBytes, Fork.class);

          byte[] cleanFork = serialize(Fork.CLEAN);

          switch (receivedFork) {
            case REQUEST -> {
              if (status.source == leftPhilosopherIndex()) {
                if (leftFork == Fork.DIRTY) {

                  leftFork = null;
                  log("Sending my fork to my left (%d) neighbour", leftPhilosopherIndex());

                  MPI.COMM_WORLD.Send(cleanFork, 0, cleanFork.length, MPI.BYTE, leftPhilosopherIndex(), 0);
                  break;
                }

                forkRequests[1] = true;
                log("Logging request for my left fork");
              } else if (status.source == rightPhilosopherIndex()) {
                if (rightFork == Fork.DIRTY) {

                  rightFork = null;
                  log("Sending my fork to my right (%d) neighbour", rightPhilosopherIndex());

                  MPI.COMM_WORLD.Send(cleanFork, 0, cleanFork.length, MPI.BYTE, rightPhilosopherIndex(), 0);
                  break;
                }

                forkRequests[0] = true;
                log("Logging request for my right fork");
              }
            }
            case DIRTY, CLEAN -> {
              String sourceNeighbourName = "-";
              if (status.source == rightPhilosopherIndex()) {
                rightFork = receivedFork;
                sourceNeighbourName = "right";
              } else if (status.source == leftPhilosopherIndex()) {
                leftFork = receivedFork;
                sourceNeighbourName = "left";
              }

              log("Got a %s fork from my %s neighbour!", receivedFork == Fork.DIRTY ? "dirty" : "clean", sourceNeighbourName);

              synchronized (this) {
                notify();
              }
            }
            default -> log("Received a spoon.");
          }
        }
      }
    }, "Listen");

    checkForkRequests.start();
  }

  private void initializePhilosopher(String[] args) {
    MPI.Init(args);

    currentIndex = MPI.COMM_WORLD.Rank();
    numberOfPhilosophers = MPI.COMM_WORLD.Size();

    initializeForks();

    log("Philosopher #%d started.", currentIndex);
  }

  /**
   * Depending on the currentIndex a certain number of forks is defined.
   */
  private void initializeForks() {
    if (currentIndex == 0) {
      // First philosopher gets both forks
      leftFork = Fork.DIRTY;
      rightFork = Fork.DIRTY;
      return;
    }

    if (currentIndex == numberOfPhilosophers - 1) {
      // Last philosopher has no forks because his fork is with the first philosopher
      return;
    }

    leftFork = Fork.DIRTY;
  }

  /**
   * Check if eating conditions are met: philosopher must have both forks.
   *
   * @return whether philosopher can eat
   */
  protected boolean canEat() {
    return (leftFork == Fork.DIRTY || leftFork == Fork.CLEAN) && (rightFork == Fork.DIRTY || rightFork == Fork.CLEAN);
  }

  /**
   * Utility method that appends philosopher details to a message before it is logged to the console.
   *
   * @param message Message to be logged (may include formatting symbols)
   * @param options Formatting options
   */
  protected void log(String message, Object... options) {
    String formattedMessage = String.format(message, options).indent(currentIndex * 6);
    String threadName = Thread.currentThread().getName();

    String hasLeftFork = leftFork == Fork.DIRTY ? "d" : leftFork == Fork.CLEAN ? "c" : "-";
    String hasRightFork = rightFork == Fork.DIRTY ? "d" : rightFork == Fork.CLEAN ? "c" : "-";

    String hasLeftForkRequest = forkRequests[1] ? "R" : " ";
    String hasRightForkRequest = forkRequests[0] ? "R" : " ";

    System.out.printf("[%d/%d @ %12s] forks: (%s %s) reqs: (%s %s) %s",
            currentIndex, numberOfPhilosophers - 1, threadName, hasLeftFork, hasRightFork, hasLeftForkRequest, hasRightForkRequest, formattedMessage);
  }


  /**
   * Utility method to get index of previous philosopher in cycle (right hand side if they were to sit at a round table).
   *
   * @return Integer index of previous philosopher (right one in the circle).
   */
  protected int rightPhilosopherIndex() {
    if (currentIndex == 0) {
      return numberOfPhilosophers - 1;
    }

    return (currentIndex - 1);
  }


  /**
   * Utility method to get index of next philosopher in cycle (left hands side if they were to sit at a round table).
   *
   * @return Integer index of previous philosopher (left one in the circle).
   */
  protected int leftPhilosopherIndex() {
    return (currentIndex + 1) % numberOfPhilosophers;
  }

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

}
