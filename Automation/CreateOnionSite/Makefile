CC = gcc
EXEC_FILE = main

.PHONY: default gcc tcc exe run wine clean

default: compile run

# Compilation
compile: main.o sys_call.o onion_path.o check_torrc.o edit_readme.o check_main_dir.o 
	gcc -o $(EXEC_FILE) main.o sys_call.o onion_path.o check_torrc.o edit_readme.o check_main_dir.o

# Run the program
run:
	./$(EXEC_FILE)

# Clear all files
clean:
	rm -rf $(EXEC_FILE) *.o *.i *.s *.exe

# C_file to Object_file
main.o: main.c
	gcc -c -o main.o -save-temps main.c
sys_call.o: sys_call.c
	gcc -c -o sys_call.o -save-temps sys_call.c
onion_path.o: onion_path.c
	gcc -c -o onion_path.o -save-temps onion_path.c
check_torrc.o: check_torrc.c
	gcc -c -o check_torrc.o -save-temps check_torrc.c
edit_readme.o: edit_readme.c
	gcc -c -o edit_readme.o -save-temps edit_readme.c
check_main_dir.o: check_main_dir.c
	gcc -c -o check_main_dir.o -save-temps check_main_dir.c

# Error if C_file was not found
main.c: 
	$(error Error: main.c was not found)
sys_call.c: 
	$(error Error: sys_call.c was not found)
onion_path.c: 
	$(error Error: onion_path.c was not found)
check_torrc.c:
	$(error Error: check_torrc.c was not found)
edit_readme.c:
	$(error Error: edit_readme.c was not found)
check_main_dir.c:
	$(error Error: check_main_dir.c was not found)
