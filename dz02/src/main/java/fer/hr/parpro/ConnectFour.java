package fer.hr.parpro;

import mpi.MPI;

public class ConnectFour {

  private static int currentIndex, numberOfPhilosophers;

  public static void main(String[] args) {
    MPI.Init(args);

    currentIndex = MPI.COMM_WORLD.Rank();
    numberOfPhilosophers = MPI.COMM_WORLD.Size();


  }

}
