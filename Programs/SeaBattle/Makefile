CC = gcc
EXEC_FILE = main

EXTEN_FILES = *.o *.i *.s
OBJECT_FILES = main.o game_menu.o setting.o actions.o extern_data.o

.PHONY: default build clean run

default: build run

build: $(OBJECT_FILES)
	$(CC) -o $(EXEC_FILE) $(OBJECT_FILES)
clean:
	rm -rf $(EXEC_FILE) $(EXTEN_FILES)
run:
	./$(EXEC_FILE)

main.o: main.c
	$(CC) -c -o main.o -save-temps main.c
game_menu.o: game_menu.c
	$(CC) -c -o game_menu.o -save-temps game_menu.c
setting.o: setting.c
	$(CC) -c -o setting.o -save-temps setting.c
actions.o: actions.c
	$(CC) -c -o actions.o -save-temps actions.c
extern_data.o: extern_data.c
	$(CC) -c -o extern_data.o -save-temps extern_data.c

main.c: 
	$(error Error: main.c was not found)
game_menu.c:
	$(error Error: game_menu.c was not found)
setting.c:
	$(error Error: setting.c was not found)
actions.c:
	$(error Error: actions.c was not found)
extern_data.c:
	$(error Error: extern_data.c was not found)
