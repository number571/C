#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>

#define BUFF 512

typedef enum {false, true} bool;

int main (void) {
    const int conn = socket(AF_INET, SOCK_STREAM, 0);
    if (conn < 0) {
        fprintf(stderr, "Error: socket\n");
        return 1;
    }

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(8080);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    if (connect(conn, (struct sockaddr *)&addr, sizeof(addr))) {
        fprintf(stderr, "Error: connect\n");
        return 2;
    }

    char buffer[BUFF];
    char *p = buffer;

    for (unsigned i = 0; (*p++ = getchar()) != '\n' && i < BUFF; ++i);
    *(p - 1) = '\0';

    send(conn, buffer, BUFF, 0);
    recv(conn, buffer, BUFF, 0);

    printf("%s\n", buffer);
    close(conn);

    return 0;
}
