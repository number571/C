#include <stdio.h>
#include <threads.h>

// gcc main.c -pthread -o main

#define COUNT 100000

int counter = 0;
mtx_t mtx;

void incFunc(void) {
    for (size_t i = 0; i < COUNT; ++i) {
        mtx_lock(&mtx);
        ++counter;
        mtx_unlock(&mtx);
    }
}

void decFunc(void) {
    for (size_t i = 0; i < COUNT; ++i) {
        mtx_lock(&mtx);
        --counter;
        mtx_unlock(&mtx);
    }
}

int main(void) {
    int code = mtx_init(&mtx, mtx_plain);
    if (code != thrd_success) {
        return 1;
    }
    thrd_t th1, th2;
    code = thrd_create(&th1, (thrd_start_t)incFunc, NULL); // NULL - args to function
    if (code != thrd_success) {
        return 2;
    }
    code = thrd_create(&th2, (thrd_start_t)decFunc, NULL); // NULL - args to function
    if (code != thrd_success) {
        return 3;
    }
    thrd_join(th1, NULL); // NULL - result code function
    thrd_join(th2, NULL); // NULL - result code function
    printf("%d\n", counter);
    mtx_destroy(&mtx);
    return 0;
}
