#pragma once

#define CHECK_MODE(x) (x == 0)?"UNREADABLE":(x == 1)?"OVERWRITTEN":"READABLE"
#define CHECK_EXIST(x) x?"READABLE":"OVERWRITTEN"

#define QUAN 7
#define BUFF 512

#define UNREADABLE  0
#define OVERWRITTEN 1
#define READABLE    2

#define WWW_PATH "/var/www/"
#define ONION_PATH "/var/www/onion/"
#define HTML_FILE_PATH "/var/www/onion/index.html"

#define MAIN_DIR "/var/lib/tor/onion/"
#define HOST_FILE "/var/lib/tor/onion/hostname"
#define KEY_FILE "/var/lib/tor/onion/private_key"

#define TORRC_PATH "/etc/tor/torrc"
#define README_PATH "README.txt"

#define HIDDEN_SERVICE_DIR "HiddenServiceDir /var/lib/tor/onion"
#define HIDDEN_SERVICE_PORT "HiddenServicePort 80 127.0.0.1:80"

typedef enum {false, true} bool;

struct List { char mode; char *path; };
struct Data { bool exist; char *string; };
