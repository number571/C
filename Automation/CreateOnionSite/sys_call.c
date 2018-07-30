#include "library.h"

#include <stdlib.h>
#include <unistd.h>

extern void start_tor_service (void);
extern void run_server (void);

extern void run_server (void) {
	chdir(ONION_PATH);
	system("python3 -m http.server 80");
}

extern void start_tor_service (void) {
    system("systemctl start tor.service");
    system("systemctl restart tor.service");
    sleep(1);
}
