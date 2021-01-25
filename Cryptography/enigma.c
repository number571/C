#include <stdio.h>
#include <string.h>

#define ROTNUM 3
#define SWTNUM 6
#define ALPSIZ 26

// A-Z, B-Y, C-X, ...
static char reflector[ALPSIZ+1] = {
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
};
// A-F, B-E, C-D, ...
static char switches[SWTNUM+1] = {
    "ABCDEF",
};
static char rotors[ROTNUM][ALPSIZ+1] = {
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
};
static int shfrot[ROTNUM] = {
    1, 2, 3,
};

extern char *encrypt(char *output, const char *input, int size);
static int find_index(char *rotor, char ch, int size);
static void shift_rotor(char *rotor, int size);

int main(int argc, char const *argv[]) {
    int len;
    char buffer[BUFSIZ];
    for (int i = 1; i < argc; ++i) {
        len = strlen(argv[i]);
        encrypt(buffer, argv[i], len);
        buffer[len] = '\0';
        printf("%s", buffer);
    }
    printf("\n");
    return 0;
}

extern char *encrypt(char *output, const char *input, int size) {
    int swt, temp;

    for (int i = 1; i < size+1; ++i) {
        temp = input[i-1];

        // Switches
        swt = find_index(switches, temp, SWTNUM);
        if (swt != -1) {
            temp = switches[SWTNUM - swt - 1];
        }

        // Rotors encrypt ->
        for (int j = 0; j < ROTNUM; ++j) {
            temp = rotors[j][temp - 'A'];
        }
        
        // Reflector
        temp = ALPSIZ - 1 - find_index(reflector, temp, ALPSIZ);

        // Rotors encrypt <-
        for (int j = ROTNUM-1; j > -1; --j) {
            temp = find_index(rotors[j], temp + 'A', ALPSIZ);
        }

        // Switches
        temp += 'A';
        swt = find_index(switches, temp, SWTNUM);
        if (swt != -1) {
            temp = switches[SWTNUM - swt - 1];
        }

        output[i-1] = temp;

        // Shift rotors
        for (int j = 0; j < ROTNUM; ++j) {
            if (i % shfrot[j] == 0) {
                shift_rotor(rotors[j], ALPSIZ);
            }
        }
    }
    return output;
}

static int find_index(char *rotor, char ch, int size) {
    for (int j = 0; j < size; ++j) {
        if (rotor[j] == ch) {
            return j;
        }
    }
    return -1;
}

static void shift_rotor(char *rotor, int size) {
    char temp = rotor[size-1];
    for (int i = size-1; i > 0; --i) {
        rotor[i] = rotor[i-1];
    }
    rotor[0] = temp;
}
