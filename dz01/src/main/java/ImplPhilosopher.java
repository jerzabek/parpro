import mpi.MPI;

/**
 * N Dining philosophers problem task for Parallel programming laboratory exercise 01.
 * Faculty of Electrical Engineering and Computing 2023.
 *
 * @author Ivan Jeržabek
 */
public class ImplPhilosopher extends Philosopher {

  public ImplPhilosopher(String[] args) {
    super(args);
  }

  public static void main(String[] args) {
    Thread.currentThread().setName("Main");

    Philosopher philosopher = new ImplPhilosopher(args);

    int i = 0;
    do {
      // Philosophers lifecycle
      philosopher.think();

      philosopher.acquireForks();

      philosopher.eat();
    } while (++i < 30); // Philosophers are mortal


    philosopher.log("I die now.");

    MPI.Finalize();
  }

  public void eat() {
    leftFork = Fork.DIRTY;
    rightFork = Fork.DIRTY;

    log("I have eaten. Yum.");

    if (!forkRequests[0] && !forkRequests[1]) {
      return;
    }


    byte[] cleanFork = serialize(Fork.CLEAN);

    if (forkRequests[1]) {
      leftFork = null;
      log("Sending my left fork to my left (%d) neighbour", leftPhilosopherIndex());

      MPI.COMM_WORLD.Send(cleanFork, 0, cleanFork.length, MPI.BYTE, leftPhilosopherIndex(), 0);

      forkRequests[1] = false;
    }

    if (forkRequests[0]) {
      rightFork = null;
      log("Sending my right fork to my right (%d) neighbour", rightPhilosopherIndex());

      MPI.COMM_WORLD.Send(cleanFork, 0, cleanFork.length, MPI.BYTE, rightPhilosopherIndex(), 0);

      forkRequests[0] = false;
    }
  }

  public void acquireForks() {
    if (canEat()) {
      log("I have both forks. I will eat now.");
      return;
    }

    log("I am looking for forks.");

    byte[] requestFork = serialize(Fork.REQUEST);

    if (rightFork == null) {
      log("Sending request to right neighbour %d", rightPhilosopherIndex());

      MPI.COMM_WORLD.Send(requestFork, 0, requestFork.length, MPI.BYTE, rightPhilosopherIndex(), 0);
    }

    if (leftFork == null) {
      log("Sending request to left neighbour %d", leftPhilosopherIndex());
      MPI.COMM_WORLD.Send(requestFork, 0, requestFork.length, MPI.BYTE, leftPhilosopherIndex(), 0);
    }


    do {
      try {
        synchronized (this) {
          this.wait(10000);
        }
      } catch (InterruptedException e) {
        // May be interrupted if a neighbour responds with a fork
      }
    } while (!canEat());

    log("I have both forks. I will eat now.");
  }

  /**
   * For a random amount of milliseconds the philosopher will ponder.
   * Amount of time will be within range [MIN_THINK_DURATION, MAX_THINK_DURATION]
   */
  public void think() {
    int durationOfThinking = MIN_THINK_DURATION + (int) (Math.random() * MAX_THINK_DURATION);

    log("I am thinking for the next %d milliseconds.", durationOfThinking);

    try {
      Thread.sleep(durationOfThinking);
    } catch (InterruptedException e) {
      throw new RuntimeException(e);
    }

    log("I have thought.");
  }

}
