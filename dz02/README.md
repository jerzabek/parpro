# Parallel programming

## FER 2022./2023. - Homework 02

### Connect four

MS-MPI and MPJ Must be installed to run this program.
Instructions: http://mpjexpress.org/guides.html

## Compile:

### Unix:

```bash
javac -cp "$MPJ_HOME/lib/mpj.jar" -d "target/classes" src/main/java/*.java
```

### Windows:

```sh
javac -cp "%MPJ_HOME%/lib/mpj.jar" -d "target/classes" src/main/java/*.java
```

## Run:

Due to MPJ Express not supporting terminal input through STDIN a workaround is used - you must communicate to the master
process via a TCP connection on the port 6868.
This can be achieved using the `telnet` utility.

Additionally, in order to evaluate the performance by changing the number of processors that the MPI processes may run
on,
the `start` command was used on a Windows environment. The same task may be achieved on UNIX systems using `taskset`.

```shell
start "ParPro Lab 02 Master" /affinity F mpjrun.bat -np 11 -cp "full\path\to\target\classes" ConnectFour
```

The argument `/affinity F` indicates which CPU cores will be assigned to this program. The value represents a bitmask in
hexadecimal format where each bit represents one core.

- Core 1: `/affinity 1`
- Core 2: `/affinity 2`
- Cores 1 and 2: `/affinity 3` (because 1 + 2 = 3)
- Core 3: `/affinity 4`
- Cores 1, 2, and 3: `/affinity 7` (because 1 + 2 + 4 = 7)
- Cores 1, 2, 3, and 4: `/affinity F` (because 1 + 2 + 4 + 8 = 15, which is F in hexadecimal)
- Cores 1..8: `/affinity FF` (because 1 + 2 + 4 + 8 + 16 + 32 + 64 + 128 = 255, which is FF in hexadecimal)

To choose a move the user must open a `telnet` session as follows:

```shell
telnet localhost 6868
```

Connection will be logged in the main terminal where MPJ is running, user must simply type one integer [1-7] to choose a
column, then press enter.