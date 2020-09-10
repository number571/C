#ifdef __linux__
	#include <unistd.h>
	#include <arpa/inet.h>
#elif __WIN32
	#include <winsock2.h>
#else
	#warning "net.h: platform not supported"
#endif

#if defined(__linux__) || defined(__WIN32)

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef enum error_t {
	WINSOCK_ERR = -1,
	SOCKET_ERR  = -2,
	SETOPT_ERR  = -3,
	PARSE_ERR   = -4,
	BIND_ERR    = -5,
	LISTEN_ERR  = -6,
	CONNECT_ERR = -7,
} error_t;

#include "net.h"

static int8_t _parse_address(char *address, char *ipv4, char *port);

// 127.0.0.1:8080
extern int listen_net(char *address) {
#ifdef __WIN32
	WSADATA wsa;
	if (WSAStartup(MAKEWORD(2,2), &wsa) != 0) {
		return WINSOCK_ERR;
	}
#endif
	int listener = socket(AF_INET, SOCK_STREAM, 0);
	if (listener < 0) {
		return SOCKET_ERR;
	}
#ifdef __linux__
	if (setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &(int){1}, sizeof(int)) < 0) {
		return SETOPT_ERR;
	}
#else
	if (setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &(char){1}, sizeof(char)) < 0) {
		return SETOPT_ERR;
	}
#endif
	char ipv4[16];
	char port[6];
	if (_parse_address(address, ipv4, port) != 0) {
		return PARSE_ERR;
	}
	struct sockaddr_in addr;
	addr.sin_family = AF_INET;
	addr.sin_port = htons(atoi(port));
	addr.sin_addr.s_addr = inet_addr(ipv4);
	if (bind(listener, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
		return BIND_ERR;
	}
	if (listen(listener, SOMAXCONN) != 0) {
		return LISTEN_ERR;
	}
	return listener;
}

extern int accept_net(int listener) {
	return accept(listener, NULL, NULL);
}

extern int connect_net(char *address) {
#ifdef __WIN32
	WSADATA wsa;
	if (WSAStartup(MAKEWORD(2,2), &wsa) != 0) {
		return WINSOCK_ERR;
	}
#endif
	int conn = socket(AF_INET, SOCK_STREAM, 0);
	if (conn < 0) {
		return SOCKET_ERR;
	}
	char ipv4[16];
	char port[6];
	if (_parse_address(address, ipv4, port) != 0) {
		return PARSE_ERR;
	}
	struct sockaddr_in addr;
	addr.sin_family = AF_INET;
	addr.sin_port = htons(atoi(port));
	addr.sin_addr.s_addr = inet_addr(ipv4);
	if (connect(conn, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
		return CONNECT_ERR;
	}
	return conn;
}

extern int close_net(int conn) {
#ifdef __linux__
	return close(conn);
#elif __WIN32
	return closesocket(conn);
#endif
}

extern int send_net(int conn, char *buffer, size_t size) {
	return send(conn, buffer, (int)size, 0);
}

extern int recv_net(int conn, char *buffer, size_t size) {
	return recv(conn, buffer, (int)size, 0);
}

static int8_t _parse_address(char *address, char *ipv4, char *port) {
	size_t i = 0, j = 0;
	for (; address[i] != ':'; ++i) {
		if (address[i] == '\0') {
			return 1;
		}
		if (i >= 15) {
			return 2;
		}
		ipv4[i] = address[i];
	}
	ipv4[i] = '\0';
	for (i += 1; address[i] != '\0'; ++i, ++j) {
		if (j >= 5) {
			return 3;
		}
		port[j] = address[i];
	}
	port[j] = '\0';
	return 0;
}

#endif /* defined(__linux__) || defined(__WIN32) */
