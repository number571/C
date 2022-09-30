#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// ENCODER
typedef struct encoder_t {
    uint8_t size_alph;
    uint8_t *alphabet;
} encoder_t;

extern encoder_t *encoder_new(uint8_t size_alph);
extern void encoder_free(encoder_t *encoder);

extern void encoder_set_alphabet(encoder_t *encoder, uint8_t *alphabet);

extern uint8_t encoder_encode(encoder_t *encoder, uint8_t ch, int *found);
extern uint8_t encoder_decode(encoder_t *encoder, uint8_t code, int *valid);

// ENIGMA
typedef struct enigma_s {
    uint64_t counter;
    uint8_t size_rotor;
    uint8_t num_rotors;
    uint8_t *reflector;
    uint8_t **rotors;
} enigma_s;

extern enigma_s *enigma_new(uint8_t size_rotor, uint8_t num_rotors);
extern void enigma_free(enigma_s *enigma);

extern void enigma_set_reflector(enigma_s *enigma, uint8_t *reflector);
extern void enigma_set_rotor(enigma_s *enigma, uint8_t num, uint8_t *rotor);

extern uint8_t enigma_encrypt(enigma_s *enigma, uint8_t code, int *valid);

static void enigma_rotor_shift(enigma_s *enigma, uint8_t num);
static uint8_t enigma_rotor_find(enigma_s *enigma, uint8_t num, uint8_t code, int *valid);

// EXAMPLE FOR SMALL ALPHABET WITH ONE ROTOR
/*
    uint8_t alphabet[] = "ABCD";
    ...
    uint8_t num_rotors = 1;
    uint8_t reflector[] = {1, 0, 3, 2};
    uint8_t *rotors[] = {
        (uint8_t[]){0, 1, 2, 3},
    };
*/

// EXAMPLE FOR FULL ALPHABET WITH THREE ROTORS
/*
    uint8_t alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    ...
    uint8_t num_rotors = 3;
    uint8_t reflector[] = {25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    uint8_t *rotors[] = {
        (uint8_t[]){0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
        (uint8_t[]){20, 21, 22, 23, 24, 25, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        (uint8_t[]){7, 6, 5, 4, 3, 2, 1, 0, 24, 23, 22, 21, 20, 25, 8, 9, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10},
    };
*/

int main(void) {
    // INIT ENCODER 
    uint8_t alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    uint8_t size_alph = (uint8_t)strlen((char*)alphabet);
    encoder_t *encoder = encoder_new(size_alph);
    encoder_set_alphabet(encoder, alphabet);

    // INIT ENIGMA
    uint8_t num_rotors = 3;
    uint8_t reflector[] = {25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    uint8_t *rotors[] = {
        (uint8_t[]){0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
        (uint8_t[]){20, 21, 22, 23, 24, 25, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        (uint8_t[]){7, 6, 5, 4, 3, 2, 1, 0, 24, 23, 22, 21, 20, 25, 8, 9, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10},
    };
    enigma_s *enigma = enigma_new(size_alph, num_rotors);

    enigma_set_reflector(enigma, reflector);
    for (int i = 0; i < num_rotors; ++i) {
        enigma_set_rotor(enigma, i, rotors[i]);
    }

    // INPUT CHARS
    uint8_t enc_ch, dec_ch;
    int ch, flag;
    while(1) {
        ch = getchar();

        // STOP IF CHAR = '~'
        if (ch == '~') {
            putchar('\n');
            break;
        }

        // ENCODE
        enc_ch = encoder_encode(encoder, (uint8_t)ch, &flag);
        if (flag == 0) {
            putchar(ch);
            continue;
        }

        // ENCRYPT/DECRYPT
        enc_ch = enigma_encrypt(enigma, enc_ch, &flag);
        if (flag == 0) {
            // encoder put to encryption unknown code
            continue; 
        }

        // DECODE
        dec_ch = encoder_decode(encoder, enc_ch, &flag);
        if (flag == 0) {
            // enigma put to decoder unknown code
            continue;
        }

        putchar(dec_ch);
    }
    
    // FREE ENCODER AND ENIGMA
    encoder_free(encoder);
    enigma_free(enigma);

    return 0;
}

// ENCODER
extern void encoder_set_alphabet(encoder_t *encoder, uint8_t *alphabet) {
    for (int i = 0; i < encoder->size_alph; ++i) {
        encoder->alphabet[i] = alphabet[i];
    }
}

extern uint8_t encoder_encode(encoder_t *encoder, uint8_t ch, int *found) {
    for (int i = 0; i < encoder->size_alph; ++i) {
        if (encoder->alphabet[i] == ch) {
            *found = 1;
            return i; 
        }
    }
    *found = 0;
    return 0;
}

extern uint8_t encoder_decode(encoder_t *encoder, uint8_t code, int *valid) {
    if (code >= encoder->size_alph) {
        *valid = 0;
        return 0;
    }
    *valid = 1;
    return encoder->alphabet[code];
}

extern void encoder_free(encoder_t *encoder) {
    free(encoder->alphabet);
    free(encoder);
}

// ENIGMA
extern enigma_s *enigma_new(uint8_t size_rotor, uint8_t num_rotors) {
    enigma_s *enigma = (enigma_s*)malloc(sizeof(enigma_s));
    if (enigma == NULL) {
        return NULL ;
    }

    enigma->size_rotor = size_rotor;
    enigma->num_rotors = num_rotors;
    enigma->counter = 0;

    enigma->reflector = (uint8_t*)malloc(sizeof(uint8_t)*size_rotor);
    enigma->rotors = (uint8_t**)malloc(sizeof(uint8_t*)*num_rotors);

    for (int i = 0; i < num_rotors; ++i) {
        enigma->rotors[i] = (uint8_t*)malloc(sizeof(uint8_t)*size_rotor);
    }

    return enigma;
}

extern void enigma_free(enigma_s *enigma) {
    for (int i = 0; i < enigma->num_rotors; ++i) {
        free(enigma->rotors[i]) ;
    }
    free(enigma->rotors);
}

extern void enigma_set_reflector(enigma_s *enigma, uint8_t *reflector) {
    for (int i = 0; i < enigma->size_rotor; ++i) {
        enigma->reflector[i] = reflector[i];
    }
}

extern void enigma_set_rotor(enigma_s *enigma, uint8_t num, uint8_t *rotor) {
    for (int i = 0; i < enigma->size_rotor; ++i) {
        enigma->rotors[num][i] = rotor[i];
    }
}

extern encoder_t *encoder_new(uint8_t size_alph) {
    encoder_t *encoder = (encoder_t*)malloc(sizeof(encoder_t));
    if (encoder == NULL) {
        return NULL;
    }
    encoder->size_alph = size_alph;
    encoder->alphabet = (uint8_t*)malloc(sizeof(uint8_t)*size_alph);
    return encoder;
}

extern uint8_t enigma_encrypt(enigma_s *enigma, uint8_t code, int *valid) {
    uint64_t rotor_queue;
    uint8_t new_code;

    if (code >= enigma->size_rotor) {
        *valid = 0;
        return 0;
    }

    new_code = code; 

    // code -> rotors
    for (int i = 0; i < enigma->num_rotors; ++i) {
        new_code = enigma->rotors[i][new_code];
    }

    // code -> reflector
    new_code = enigma->reflector[new_code];
    // reflector -> code

    // rotors -> code
    for (int i = enigma->num_rotors-1; i >= 0; --i) {
        new_code = enigma_rotor_find(enigma, i, new_code, valid);
        if (*valid == 0) {
            return 0;
        }
    }

    // shift rotors
    rotor_queue = 1;
    enigma->counter += 1;
    for (int i = 0; i < enigma->num_rotors; ++i) {
        if (enigma->counter % rotor_queue == 0) {
            enigma_rotor_shift(enigma, i);
        }
        rotor_queue *= enigma->size_rotor;
    }

    *valid = 1;
    return new_code;
}

static void enigma_rotor_shift(enigma_s *enigma, uint8_t num) {
    char temp = enigma->rotors[num][enigma->size_rotor-1];
    for (int i = enigma->size_rotor-1; i > 0; --i) {
        enigma->rotors[num][i] = enigma->rotors[num][i-1];
    }
    enigma->rotors[num][0] = temp;
}

static uint8_t enigma_rotor_find(enigma_s *enigma, uint8_t num, uint8_t code, int *valid) {
    for (int i = 0; i < enigma->size_rotor; ++i) {
        if (enigma->rotors[num][i] == code) {
            *valid = 1;
            return i;
        }
    }
    *valid = 0;
    return 0;
}
