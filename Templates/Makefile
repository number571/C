CC = gcc
EXEC_FILE = main
EXTEN_FILES = *.o *.i *.s *.exe

.PHONY: default gcc tcc exe run wine clean

default: $(CC) run

# Compilation
gcc: main.o
	gcc -o $(EXEC_FILE) main.o
tcc: main.o
	tcc -o $(EXEC_FILE) main.o
exe: main.c
	wine TCC/tcc.exe -o $(EXEC_FILE).exe main.c

# Run the program
run:
	./$(EXEC_FILE)
wine:
	wine $(EXEC_FILE).exe

# Clear all files
clean:
	rm -rf $(EXEC_FILE) $(EXTEN_FILES)

# C_file to Object_file
main.o: main.c
	gcc -c -o main.o -save-temps main.c

# Error if C_file was not found
main.c: 
	$(error Error: main.c was not found)
