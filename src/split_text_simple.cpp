#include <bits/stdc++.h>
#include "split_text_common.h"

std::unordered_map<std::wstring, int> str2index;
std::vector<const wchar_t*> index2str;
std::vector<int> text;
FILE *fptr;

void split_text()
{
    int token_counter = 0;
    std::wstring token;
    wint_t wc;

    while((wc = fgetwc(fptr)) != WEOF) {
        if (!iswalnum(wc)) {
            if (token.size() != 0) {
                if (str2index.find(token) == str2index.end()) {
                    str2index[token] = token_counter;
                    token_counter++;
                }

                int tok_index = str2index[token];
                text.push_back(tok_index);
                token.resize(0);
            }

            if (!iswspace(wc)) {
                std::wstring sep_token(1, wc);
                if (str2index.find(sep_token) == str2index.end()) {
                    str2index[sep_token] = token_counter;
                    token_counter++;
                }

                int tok_index = str2index[sep_token];
                text.push_back(tok_index);
            }
        }
        else {
            token += towlower(wc);
        }
    }
}

int main(int argc, char **argv)
{
    setlocale(LC_ALL, "en_US.UTF-8");

    if (argc != 4) {
        fprintf(stderr, "usage example: split_text_simple text.txt words.txt text.bin\n");
        return 1;
    }

    fptr = fopen(argv[1], "r");
    if (fptr == NULL) {
        fprintf(stderr, "can't open %s\n", argv[1]);
        return 1;
    }

    split_text();
    fclose(fptr);

    index2str.resize(str2index.size());
    for (auto &x : str2index)
        index2str[x.second] = x.first.c_str();

    write_words(argv[2], index2str);
    write_text(argv[3], text);
}
