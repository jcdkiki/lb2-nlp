#include <algorithm>
#include <bits/stdc++.h>
#include <cwchar>
#include <cwctype>
#include <stdio.h>
#include <string>

static constexpr int CONTEXT_RADIUS = 3;
static constexpr int EMBEDDING_SIZE = 500;
static constexpr double GRADIENT_FACTOR = 0.01;

struct Embedding {
    double arr[EMBEDDING_SIZE];
};

struct TokenCount {
    int index;
    int count;
};

double dot(const Embedding &a, const Embedding &b)
{
    double res = 0.0;
    for (int i = 0; i < EMBEDDING_SIZE; i++) {
        res += a.arr[i] * b.arr[i];
    }

    return res;
}

bool operator<(const TokenCount &a, const TokenCount &b)
{
    if (a.count != b.count) return a.count < b.count;
    return a.index < b.index;
}

std::unordered_map<std::wstring, int> str2index;
std::vector<const wchar_t*> index2str;
std::vector<int> unsorted_token_count;
std::vector<TokenCount> token_count;
std::vector<Embedding> embeddings;
FILE *fptr;

std::vector<int> text;

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
                    unsorted_token_count.push_back(0);
                    token_counter++;
                }

                int tok_index = str2index[token];
                text.push_back(tok_index);
                unsorted_token_count[tok_index]++;
                token.resize(0);
            }

            if (!iswspace(wc)) {
                std::wstring sep_token(1, wc);
                if (str2index.find(sep_token) == str2index.end()) {
                    str2index[sep_token] = token_counter;
                    unsorted_token_count.push_back(0);
                    token_counter++;
                }

                int tok_index = str2index[sep_token];
                text.push_back(tok_index);
                unsorted_token_count[tok_index]++;
            }
        }
        else {
            token += towlower(wc);
        }
    }
}

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

bool is_in_positive_context(int mid, int word)
{
    for (int i = mid - CONTEXT_RADIUS; i <= mid + CONTEXT_RADIUS; i++) {
        if (text[i] == word)
            return false;
    }
    return true;
}

void find_negative_context(int word, std::vector<int> &res)
{
    res.clear();
    int word_count = unsorted_token_count[word];
    int count_range = word_count / 10;

    auto left = std::lower_bound(token_count.begin(), token_count.end(), TokenCount { 0, word_count - count_range });
    auto right = std::upper_bound(token_count.begin(), token_count.end(), TokenCount { 0, word_count + count_range + 1 });
    
    for (int i = 0; i < CONTEXT_RADIUS * 8; i++) {
        if (res.size() == CONTEXT_RADIUS * 4)
            return;

        auto it = left + (rand() % (right - left));
        if (!is_in_positive_context(word, it->index) && std::find(res.begin(), res.end(), word) == res.end())
            res.push_back(it->index);
    }
}

void run_sliding_context()
{
    int context_size = CONTEXT_RADIUS * 2 + 1;
    int half = context_size / 2;
    int left = 0;
    
    std::vector<int> negative_context;

    while (left + context_size != text.size()) {
        int progress = (left + context_size) * 100 / text.size();
        int prev_progress = (left + context_size - 1) * 100 / text.size();
        if (progress / 5 != prev_progress / 5)
            wprintf(L"%d%%\n", progress);

        int mid_word = text[left + half];
        Embedding new_embedding = embeddings[mid_word];

        for (int i = 0; i < context_size; i++) {
            if (i == half) continue;
            
            double coeff = (sigmoid(dot(embeddings[text[left + i]], embeddings[mid_word])) - 1) * GRADIENT_FACTOR;
            for (int j = 0; j < EMBEDDING_SIZE; j++) {
                new_embedding.arr[j] -= embeddings[text[left + i]].arr[j] * coeff;
            }
        }

        find_negative_context(mid_word, negative_context);

        for (int i = 0; i < negative_context.size(); i++){
            double coeff = sigmoid(dot(embeddings[negative_context[i]], embeddings[mid_word])) * GRADIENT_FACTOR;
            for (int j = 0; j < EMBEDDING_SIZE; j++) {
                new_embedding.arr[j] -= embeddings[negative_context[i]].arr[j] * coeff;
            }
        }

        embeddings[mid_word] = new_embedding;
        left++;
    }
}

double randrange(double min, double max) 
{
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

int main(int argc, char **argv)
{
    setlocale(LC_ALL, "en_US.UTF-8");
    argc--; argv++;

    if (argc == 0) {
        fprintf(stderr, "specify filename\n");
        return 1;
    }

    fptr = fopen(argv[0], "r");
    
    if (fptr == NULL) {
        fprintf(stderr, "fopen failed\n");
        return 1;
    }

    wprintf(L"splitting text\n");
    split_text();

    fclose(fptr);

    index2str.resize(str2index.size());
    embeddings.resize(str2index.size());
    wprintf(L"unique words: %d\n", (int)str2index.size());

    wprintf(L"randomize embeddings\n");
    for (auto &x : embeddings) {
        for (int i = 0; i < EMBEDDING_SIZE; i++) {
            x.arr[i] = randrange(-10, 10);
        }
    }

    wprintf(L"inverting str2index\n");
    for (auto &x : str2index) {
        index2str[x.second] = x.first.c_str();
    }

    wprintf(L"sorting token_count\n");
    token_count.resize(unsorted_token_count.size());
    for (int i = 0; i < unsorted_token_count.size(); i++) {
        token_count[i].count = unsorted_token_count[i];
        token_count[i].index = i;
    }

    std::sort(token_count.begin(), token_count.end());

    wprintf(L"running sliding context\n");
    run_sliding_context();

    wprintf(L"writing data to file\n");
    
    FILE *fout = fopen("words.txt", "w");
    for (int i = 0; i < index2str.size(); i++) {
        fwprintf(fout, L"%ls\n", index2str[i]);
    }
    fclose(fout);
    
    fout = fopen("embeddings.bin", "wb");
    for (int i = 0; i < embeddings.size(); i++) {
        fwrite(embeddings[i].arr, sizeof(double) * EMBEDDING_SIZE, 1, fout);
    }
    fclose(fout);
}
