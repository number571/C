#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>

#define BUFF 512

char buffer[BUFF] = "hello, world";

int main (void) {
    const int conn = socket(AF_INET, SOCK_STREAM, 0);
    if (conn < 0) {
        fprintf(stderr, "Error: socket\n");
        return 1;
    }

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(8080);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (connect(conn, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "Error: connect\n");
        return 2;
    }

    send(conn, buffer, BUFF, 0);
    recv(conn, buffer, BUFF, 0);

    printf("%s\n", buffer);
    close(conn);

    return 0;
}
