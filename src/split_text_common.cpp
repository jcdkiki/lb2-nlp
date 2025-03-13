#include "split_text_common.h"
#include <cstdio>
#include <cstdlib>
#include <wchar.h>

void write_words(const char *filename, const std::vector<const wchar_t*> &words)
{
    FILE *fptr = fopen(filename, "w");
    if (fptr == NULL) {
        fprintf(stderr, "can't open %s\n", filename);
        exit(1);
    }
    
    for (int i = 0; i < words.size(); i++) {
        fwprintf(fptr, L"%ls\n", words[i]);
    }

    fclose(fptr);
}

void write_text(const char *filename, const std::vector<int> &text)
{
    FILE *fout = fopen(filename, "wb");
    if (fout == NULL) {
        fprintf(stderr, "can't open %s\n", filename);
        exit(1);
    }

    int text_len = text.size();
    fwrite(&text_len, sizeof(int), 1, fout);
    
    for (int i = 0; i < text.size(); i++) {
        fwrite(&text[i], sizeof(int), 1, fout);
    }
    
    fclose(fout);
}
