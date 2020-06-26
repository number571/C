CC=gcc
CFLAGS=-Wall -std=c99

# CC=i686-w64-mingw32-gcc
# LN=ld -m i386pe

FILES=main.c extclib/extclib.o

.PHONY: default build run
default: build run

build: $(FILES)
	$(CC) $(CFLAGS) $(FILES) -o main # add '-lws2_32' if platform 'windows'
run: main
	./main
