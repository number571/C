CC = gcc
EXEC_FILE = main
EXTEN_FILES = *.o *.i *.s
OBJECT_FILES = main.o sys_call.o onion_path.o check_torrc.o edit_readme.o check_main_dir.o extern_data.o

.PHONY: default build run clean

default: build

# Compilation
build: $(OBJECT_FILES)
	$(CC) -o $(EXEC_FILE) $(OBJECT_FILES)

# Run the program
run:
	./$(EXEC_FILE)

# Clear all files
clean:
	rm -rf $(EXEC_FILE) $(EXTEN_FILES)

# C_file to Object_file
main.o: main.c
	$(CC) -c -o main.o -save-temps main.c
sys_call.o: sys_call.c
	$(CC) -c -o sys_call.o -save-temps sys_call.c
onion_path.o: onion_path.c
	$(CC) -c -o onion_path.o -save-temps onion_path.c
check_torrc.o: check_torrc.c
	$(CC) -c -o check_torrc.o -save-temps check_torrc.c
edit_readme.o: edit_readme.c
	$(CC) -c -o edit_readme.o -save-temps edit_readme.c
check_main_dir.o: check_main_dir.c
	$(CC) -c -o check_main_dir.o -save-temps check_main_dir.c
extern_data.o: extern_data.c
	$(CC) -c -o extern_data.o -save-temps extern_data.c

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
extern_data.c:
	$(error Error: extern_data.c was not found)
