CC = gcc
EXEC_FILE = main

EXTEN_FILES = *.o *.i *.s
OBJECT_FILES = main.o platform.o menu.o action.o game.o

.PHONY: default compile clean run

default: compile run

compile: $(OBJECT_FILES)
	$(CC) -o $(EXEC_FILE) $(OBJECT_FILES)
clean:
	rm -rf $(EXEC_FILE) $(EXTEN_FILES)
run:
	./$(EXEC_FILE)

main.o: main.c
	$(CC) -c -o main.o -save-temps main.c
platform.o: platform.c
	$(CC) -c -o platform.o -save-temps platform.c
menu.o: menu.c
	$(CC) -c -o menu.o -save-temps menu.c
action.o: action.c
	$(CC) -c -o action.o -save-temps action.c
game.o: game.c
	$(CC) -c -o game.o -save-temps game.c
