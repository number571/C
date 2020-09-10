# CC=gcc
# CFLAGS=-Wall -std=c99

CC=i686-w64-mingw32-gcc
LN=ld -m i386pe

FILES=main.c extclib/extclib.o

.PHONY: default build run
default: build run

build: $(FILES)
# 	$(CC) $(CFLAGS) $(FILES) -o main
	$(CC) $(CFLAGS) $(FILES) -o main.exe -lws2_32
run: main
# 	./main
	wine main.exe
