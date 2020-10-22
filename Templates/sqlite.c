// gcc main.c sqlite3.c -o main -lpthread -ldl

#include <stdio.h>
#include "sqlite3.h"

int callback(void *arg, int num, char **elem, char **colname);

int main(void) {
    int rc;
    sqlite3 *db;

    rc = sqlite3_open("database.db", &db);
    if (rc != SQLITE_OK) {
        return 1;
    }

    rc = sqlite3_exec(db, 
        "CREATE TABLE IF NOT EXISTS users ("
            "id INTEGER,"
            "name VARCHAR(44) UNIQUE,"
            "PRIMARY KEY (id)"
        ");",
    NULL, NULL, NULL);
    if (rc != SQLITE_OK) {
        return 2;
    }

    char buffer[BUFSIZ];
    char *names[3] = {"Alice", "Bob", "Eve"};

    for (size_t i = 0; i < 3; ++i) {
        snprintf(buffer, BUFSIZ, "INSERT INTO users (name) VALUES ('%s');", names[i]);
        sqlite3_exec(db, buffer, NULL, NULL, NULL);
    }

    rc = sqlite3_exec(db, "SELECT * FROM users;", callback, NULL, NULL);
    if (rc != SQLITE_OK) {
        return 4;
    }

    sqlite3_close(db);
    return 0;
}

int callback(void *arg, int num, char **elem, char **colname) {
    printf("Column [\n");
    for (int i = 0; i < num; ++i) {
        printf("\t%s = %s\n", colname[i], elem[i]);
    }
    printf("]\n\n");
    return 0;
}
