#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INFINIT 10000000
#define UNKNOWN -1
#define SELFNOD 0
#define RELSIZE 6

typedef enum node_t {
    A, B, C, D, E, F
} node_t;

typedef struct relation_t {
    _Bool used;
    int paths[RELSIZE];
} relation_t;

relation_t *new_relation(int paths[RELSIZE]);
void init_rels(relation_t *rels[RELSIZE]);
void free_rels(relation_t *rels[RELSIZE]);

int dijkstra(relation_t *rels[RELSIZE], node_t start, node_t finish) {
    int results[RELSIZE] = {0};
    for (size_t i = 0; i < RELSIZE; ++i) {
        results[i] = INFINIT;
    }
    results[start] = 0;
    // enumeration of nodes
    for (size_t i = 0; i < RELSIZE; ++i) {
        // enumeration of node relations
        for (size_t j = 0; j < RELSIZE; ++j) {
            if (rels[i]->paths[j] != UNKNOWN && !rels[i]->used) {
                if (results[j] < results[i] + rels[i]->paths[j]) {
                    continue;
                }
                results[j] = results[i] + rels[i]->paths[j];
            }
        }
        rels[i]->used = 1;
    }
    return results[finish];
}

int main(void) {
    relation_t *rels[RELSIZE];
    init_rels(rels);
    printf("%d\n", dijkstra(rels, A, E));
    free_rels(rels);
    return 0;
}

relation_t *new_relation(int paths[RELSIZE]) {
    relation_t *rel = (relation_t*)malloc(sizeof(relation_t));
    rel->used = 0;
    memcpy(rel->paths, paths, RELSIZE*sizeof(int));
    return rel;
}

// EXAMPLE FROM: https://ru.wikipedia.org/wiki/%D0%90%D0%BB%D0%B3%D0%BE%D1%80%D0%B8%D1%82%D0%BC_%D0%94%D0%B5%D0%B9%D0%BA%D1%81%D1%82%D1%80%D1%8B
void init_rels(relation_t *rels[RELSIZE]) {
    rels[A] = new_relation((int[]){
        [A] = SELFNOD,
        [B] = 7,
        [C] = 9,
        [D] = UNKNOWN,
        [E] = UNKNOWN,
        [F] = 14,
    });

    rels[B] = new_relation((int[]){
        [A] = 7,
        [B] = SELFNOD,
        [C] = 10,
        [D] = 15,
        [E] = UNKNOWN,
        [F] = UNKNOWN,
    });

    rels[C] = new_relation((int[]){
        [A] = 9,
        [B] = 10,
        [C] = SELFNOD,
        [D] = 11,
        [E] = UNKNOWN,
        [F] = 2,
    });

    rels[D] = new_relation((int[]){
        [A] = UNKNOWN,
        [B] = 15,
        [C] = 11,
        [D] = SELFNOD,
        [E] = 6,
        [F] = UNKNOWN,
    });

    rels[E] = new_relation((int[]){
        [A] = UNKNOWN,
        [B] = UNKNOWN,
        [C] = UNKNOWN,
        [D] = 6,
        [E] = SELFNOD,
        [F] = 9,
    });

    rels[F] = new_relation((int[]){
        [A] = 14,
        [B] = UNKNOWN,
        [C] = 2,
        [D] = UNKNOWN,
        [E] = 9,
        [F] = SELFNOD,
    });
}

void free_rels(relation_t *rels[RELSIZE]) {
    for (size_t i = 0; i < RELSIZE; ++i) {
        free(rels[i]);
    }
}
