CC=gcc
LN=ld

CFLAGS=-Wall -std=c99

HEADERS=stack.h hashtab.h tree.h type.h
SOURCES=stack.c hashtab.c tree.c type.c
OBJECTS=stack.o hashtab.o tree.o type.o

.PHONY: default build link clean
default: build link
	
build: $(HEADERS) $(SOURCES)
	$(CC) $(CFLAGS) -c $(SOURCES)
link: $(OBJECTS)
	$(LN) -r $(OBJECTS) -o extclib.o
clean:
	rm -f *.o *.i *.s
