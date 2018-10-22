#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>

#define BUFF 512

typedef enum {false, true} bool;

int main (void) {
    int listener = socket(AF_INET, SOCK_STREAM, 0);
    if (listener < 0) {
        fprintf(stderr, "Error: socket\n");
        return 1;
    }

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(8080);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(listener, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "Error: bind\n");
        return 2;
    }

    printf("Server is listening ...\n");

    char buffer[BUFF];
    listen(listener, 1);

    while (true) {
        int conn = accept(listener, NULL, NULL);

        if (conn < 0) {
            fprintf(stderr, "Error: accept\n");
            return 3;
        }

        while (true) {
            int length = recv(conn, buffer, BUFF, 0);
            if (length <= 0) break;
            for (char *p = buffer; *p != '\0'; ++p)
                *p = toupper(*p);
            send(conn, buffer, BUFF, 0);
        }

        close(conn);
    }

    return 0;
}
