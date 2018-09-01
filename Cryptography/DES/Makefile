CC = gcc
EXTEN_FILES = *.o *.i *.s *.exe

.PHONY: default build run clean

default: build run

build: main.o des.o
	$(CC) -o main main.o des.o

run: main
	./main

clean:
	rm -rf main $(EXTEN_FILES)

main.o: main.c
	$(CC) -c -o main.o -save-temps main.c

des.o: des.c
	$(CC) -c -o des.o -save-temps des.c

main.c:
	$(error Error: main.c was not found)

des.c:
	$(error Error: des.c was not found)
