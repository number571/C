#include <stdio.h>
#include <string.h>

#define ALPSIZ 26

static char reflector[ALPSIZ+1] = {
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
};
static char rotors[3][ALPSIZ+1] = {
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
};

char *encrypt(char *output, const char *input, int size);
int find_index(char *rotor, char ch);
void shift_rotor(char *rotor, int size);

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

char *encrypt(char *output, const char *input, int size) {
    int rot1, rot2, rot3;
    for (int i = 1; i < size+1; ++i) {
        rot1 = input[i-1];

        rot1 = rotors[0][rot1 - 'A'];
        rot2 = rotors[1][rot1 - 'A'];
        rot3 = rotors[2][rot2 - 'A'];

        rot3 = ALPSIZ - 1 - find_index(reflector, rot3);
        rot3 = find_index(reflector, rot3 + 'A');

        rot3 = find_index(rotors[2], rot3 + 'A');
        rot2 = find_index(rotors[1], rot3 + 'A');
        rot1 = find_index(rotors[0], rot2 + 'A');

        output[i-1] = rot1 + 'A';

        if (i % 1 == 0) {
            shift_rotor(rotors[0], ALPSIZ);
        }
        if (i % 2 == 0) {
            shift_rotor(rotors[1], ALPSIZ);
        }
        if (i % 3 == 0) {
            shift_rotor(rotors[2], ALPSIZ);
        }
    }
    return output;
}

int find_index(char *rotor, char ch) {
    for (int j = 0; j < ALPSIZ; ++j) {
        if (rotor[j] == ch) {
            return j;
        }
    }
    return -1;
}

void shift_rotor(char *rotor, int size) {
    char temp = rotor[size-1];
    for (int i = size-1; i > 0; --i) {
        rotor[i] = rotor[i-1];
    }
    rotor[0] = temp;
}
