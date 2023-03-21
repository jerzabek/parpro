# Parallel programming
## FER 2022./2023. - Homework 01

### N dining philosophers problem

MS-MPI and MPJ Must be installed to run this program.
Instructions: http://mpjexpress.org/guides.html

## Compile:

### Unix:
```shell
javac -cp "lib/mpj.jar" -d "target/classes" src/main/java/ImplPhilosopher.java
```

### Windows:
```shell
javac -cp "%MPJ_HOME%/lib/mpj.jar" -d "target/classes" src/main/java/ImplPhilosopher.java
```

## Run:

Depending on UNIX _(mpjrun.sh)_ or Windows _(mpjrun.sh)_ environment use different script file

```shell
mpjrun.bat -np 2 -cp target/classes ImplPhilosopher
```