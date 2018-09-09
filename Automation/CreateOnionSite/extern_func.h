extern void create_onion (void);
extern void check_torrc (void);
extern void check_main_dir (void);
extern void start_tor_service (void);
extern void edit_readme (void);
extern void run_server (void);

void activate (void) {
    create_onion();
    check_torrc();
    check_main_dir();
    start_tor_service();
    edit_readme();
    run_server();
}
