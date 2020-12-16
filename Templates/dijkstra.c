#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INFINIT 10000000
#define UNKNOWN -1
#define SELFNOD 0
#define RELSIZE 5

typedef enum node_t {
    A, B, C, D, E, F
} node_t;

int *new_relation(int paths[RELSIZE]);
void init_rels(int *rels[RELSIZE]);
void free_rels(int *rels[RELSIZE]);

static inline void _dijkstra(int *results, int **rels, int i, size_t size) {
    // enumeration of node relations
    for (size_t j = 0; j < size; ++j) {
        if (rels[i][j] != UNKNOWN) {
            int dist = results[i] + rels[i][j];
            if (results[j] < dist) {
                continue;
            }
            results[j] = dist;
        }
    }
}

int dijkstra(int **rels, node_t start, node_t finish, size_t size) {
    int results[size];
    for (size_t i = 0; i < size; ++i) {
        results[i] = INFINIT;
    }
    results[start] = 0;
    // enumeration of nodes
    for (int i = 0; i < size; ++i) {
        _dijkstra(results, rels, i, size);
    }
    for (int i = size-1; i >= 0; --i) {
        _dijkstra(results, rels, i, size);
    }
    return results[finish];
}

int main(void) {
    int *rels[RELSIZE];
    init_rels(rels);
    printf("%d\n", dijkstra(rels, A, C, RELSIZE));
    free_rels(rels);
    return 0;
}

int *new_relation(int paths[RELSIZE]) {
    int *rel = (int*)malloc(sizeof(int)*RELSIZE);
    memcpy(rel, paths, sizeof(int)*RELSIZE);
    return rel;
}

// EXAMPLE FROM: https://habr.com/ru/post/111361/
void init_rels(int *rels[RELSIZE]) {
    rels[A] = new_relation((int[]){
        [A] = SELFNOD,
        [B] = 10,
        [C] = 100,
        [D] = UNKNOWN,
        [E] = 30,
    });

    rels[B] = new_relation((int[]){
        [A] = 10,
        [B] = SELFNOD,
        [C] = UNKNOWN,
        [D] = 50,
        [E] = UNKNOWN,
    });

    rels[C] = new_relation((int[]){
        [A] = 100,
        [B] = UNKNOWN,
        [C] = SELFNOD,
        [D] = 10,
        [E] = 60,
    });

    rels[D] = new_relation((int[]){
        [A] = UNKNOWN,
        [B] = 50,
        [C] = 10,
        [D] = SELFNOD,
        [E] = 20,
    });

    rels[E] = new_relation((int[]){
        [A] = 30,
        [B] = UNKNOWN,
        [C] = 60,
        [D] = 20,
        [E] = SELFNOD,
    });
}

// EXAMPLE FROM: https://ru.wikipedia.org/wiki/%D0%90%D0%BB%D0%B3%D0%BE%D1%80%D0%B8%D1%82%D0%BC_%D0%94%D0%B5%D0%B9%D0%BA%D1%81%D1%82%D1%80%D1%8B
// void init_rels(int *rels[RELSIZE]) {
//     rels[A] = new_relation((int[]){
//         [A] = SELFNOD,
//         [B] = 7,
//         [C] = 9,
//         [D] = UNKNOWN,
//         [E] = UNKNOWN,
//         [F] = 14,
//     });

//     rels[B] = new_relation((int[]){
//         [A] = 7,
//         [B] = SELFNOD,
//         [C] = 10,
//         [D] = 15,
//         [E] = UNKNOWN,
//         [F] = UNKNOWN,
//     });

//     rels[C] = new_relation((int[]){
//         [A] = 9,
//         [B] = 10,
//         [C] = SELFNOD,
//         [D] = 11,
//         [E] = UNKNOWN,
//         [F] = 2,
//     });

//     rels[D] = new_relation((int[]){
//         [A] = UNKNOWN,
//         [B] = 15,
//         [C] = 11,
//         [D] = SELFNOD,
//         [E] = 6,
//         [F] = UNKNOWN,
//     });

//     rels[E] = new_relation((int[]){
//         [A] = UNKNOWN,
//         [B] = UNKNOWN,
//         [C] = UNKNOWN,
//         [D] = 6,
//         [E] = SELFNOD,
//         [F] = 9,
//     });

//     rels[F] = new_relation((int[]){
//         [A] = 14,
//         [B] = UNKNOWN,
//         [C] = 2,
//         [D] = UNKNOWN,
//         [E] = 9,
//         [F] = SELFNOD,
//     });
// }

void free_rels(int *rels[RELSIZE]) {
    for (size_t i = 0; i < RELSIZE; ++i) {
        free(rels[i]);
    }
}
