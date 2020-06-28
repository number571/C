CC=gcc
CFLAGS=-Wall -std=c99

FILES=main.c extclib/extclib.o

.PHONY: default build run
default: build run

build: $(FILES)
	$(CC) $(CFLAGS) $(FILES) -o main -lgmp
run: main
	./main main.sm
