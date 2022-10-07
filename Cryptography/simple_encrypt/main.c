#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

uint8_t simple_encrypt(uint8_t *in_alpha, uint8_t *out_alpha, uint8_t ch);
void shuffle_alpha(uint8_t *alpha);

int main() {
    FILE *input, *output;
    int ch;

    uint8_t input_alpha[]  = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    uint8_t output_alpha[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    // COMMENT THIS LINE FOR DECRYPT
    shuffle_alpha(output_alpha);

    input = fopen("input.txt", "r");
    output = fopen("output.txt", "w");

    fprintf(output, "%s\n\n", output_alpha);
    while((ch = fgetc(input)) != EOF) {
        fputc(simple_encrypt(input_alpha, output_alpha, toupper(ch)), output);
    }

    fclose(input);
    fclose(output);
}

uint8_t simple_encrypt(uint8_t *in_alpha, uint8_t *out_alpha, uint8_t ch) {
    int size;

    size = strlen((char*)in_alpha);
    if (size != strlen((char*)out_alpha)) {
        return '~';
    }

    for (int i = 0; i < size; ++i) {
        if (in_alpha[i] == ch) {
            return out_alpha[i];
        } 
    }

    return ch;
}

void shuffle_alpha(uint8_t *alpha) {
    uint8_t t, j;
    int size;

    size = strlen((char*)alpha);
    srand(time(NULL));

    for (int i = 0; i < size; i++) {
        j = rand() % size;
        t = alpha[j];
        alpha[j] = alpha[i];
        alpha[i] = t;
    }
}
