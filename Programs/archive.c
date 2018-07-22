#include <stdio.h>
#define UINT64 unsigned long long int

/* [Archive]:
 *      ./main archive file1 file2 file3 ...
 * [Unarchive]:
 *      ./main archive
 */

void get_length (UINT64 array[], const int argc, const char const *argv[]);
void input_length (FILE* const arch, UINT64 array[], const int argc, char const *argv[]);
void input_files (FILE* const arch, const int argc, const char const *argv[]);

void archive (FILE* const arch, const int argc, const char const *argv[]);
void unarchive (FILE* const arch);

int main (const int argc, const char const *argv[]) {
    if (argc < 2) {
        printf("Error: archive not found.\n");
        return 1;
    }

    auto FILE *arch;
    if (argc != 2) {
        arch = fopen(argv[1], "wb");
        if (arch == NULL) return 1;
        archive(arch, argc, argv);
    } else {
        arch = fopen(argv[1], "rb");
        if (arch == NULL) return 1;
        unarchive(arch);
    }

    fclose(arch);
    return 0;
} 

void unarchive (FILE* const arch) {
    auto UINT64 now_position = 0;
    auto UINT64 start_position = 0;

    auto int c;
    while ((c = getc(arch)) != EOF) {
        start_position++;
        if (c == '\n') break;
    }
    fseek(arch, 0, SEEK_SET);

    auto char filename[128];
    auto UINT64 filesize;

    auto FILE *file;
    while (fscanf(arch, "| %llu = %s |", &filesize, filename) != 0) {
        file = fopen(filename, "wb");
        if (file == NULL) break;

        printf("|File: %s = Bytes: %llu|\n", filename, filesize);

        now_position = ftell(arch);
        fseek(arch, start_position, SEEK_SET);

        start_position += filesize;
        while (filesize-- > 0)
            putc((c = getc(arch)), file);

        fseek(arch, now_position, SEEK_SET);

        fclose(file);
    }
}

void archive (FILE* const arch, const int argc, const char const *argv[]) {
    auto UINT64 save_length[argc-2];
    get_length(save_length, argc, argv);
    input_length(arch, save_length, argc, argv);
    input_files(arch, argc, argv);
}

void get_length (UINT64 array[], const int argc, const char const *argv[]) {
    auto FILE *file;
    auto unsigned char index;
    for (index = 2; index < argc; index++) {
        file = fopen(argv[index], "rb");
        if (file == NULL) continue;

        fseek(file, 0, SEEK_END);
            array[index-2] = ftell(file);
        fseek(file, 0, SEEK_SET);

        fclose(file);
    }
}

void input_length (FILE* const arch, UINT64 array[], const int argc, char const *argv[]) {
    auto unsigned char index;
    for (index = 0; index < argc-2; index++)
        fprintf(arch, "| %llu = %s |", array[index], argv[index+2]);
    fprintf(arch, "\n");
}

void input_files (FILE* const arch, const int argc, const char const *argv[]) {
    auto FILE *file;
    auto int c;
    auto unsigned char index;
    for (index = 2; index < argc; index++) {
        file = fopen(argv[index], "rb");
        if (file == NULL) continue;
        while ((c = getc(file)) != EOF)
            putc(c, arch);
        fclose(file);
    }
}
