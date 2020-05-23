#include <stdio.h>
#include <unistd.h>

int main(void) {
    printf("begin\n");
    pid_t ret;
    switch(ret = fork()) {
        case -1: // error
        break;
        case 0: // child
            printf("child\n");
        break;
        default: // parent
            printf("parent\n");
    }
    printf("end\n");
    return 0;
}
