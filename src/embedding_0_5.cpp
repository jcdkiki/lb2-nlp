#include <algorithm>
#include <bits/stdc++.h>

static constexpr int CONTEXT_RADIUS = 5;
static constexpr int EMBEDDING_SIZE = 100;
static constexpr double GRADIENT_FACTOR = 0.025;
static constexpr double NEG_FACTOR = 2;

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

int n_words;
std::vector<int> unsorted_token_count;
std::vector<TokenCount> token_count;
std::vector<Embedding> embeddings;
std::vector<int> text;

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

void find_negative_context(int word_index, std::vector<int> &res)
{
    int word = text[word_index];
    res.clear();
    int word_count = unsorted_token_count[word];
    int count_range = word_count / 10;

    auto left = std::lower_bound(token_count.begin(), token_count.end(), TokenCount { 0, word_count - count_range });
    auto right = std::upper_bound(token_count.begin(), token_count.end(), TokenCount { 0, word_count + count_range + 1 });
    
    for (int i = 0; i < CONTEXT_RADIUS * NEG_FACTOR * 2; i++) {
        if (res.size() == CONTEXT_RADIUS * NEG_FACTOR)
            return;

        auto it = left + (rand() % (right - left));
        if (!is_in_positive_context(word_index, it->index) && std::find(res.begin(), res.end(), word) == res.end())
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

        find_negative_context(left + half, negative_context);

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

void read_text(const char *filename)
{
    FILE *fptr = fopen(filename, "rb");
    
    if (fptr == NULL) {
        fprintf(stderr, "failed to open %s\n", filename);
        exit(1);
    }

    int text_len;
    fread(&text_len, sizeof(int), 1, fptr);

    text.resize(text_len);
    fread(&text[0], sizeof(int), text_len, fptr);

    fclose(fptr);
}

void get_token_count()
{
    n_words = *std::max_element(text.begin(), text.end()) + 1;
    unsorted_token_count.resize(n_words);


    for (int w : text) {
        unsorted_token_count[w]++;
    }

    token_count.resize(n_words);
    for (int i = 0; i <= n_words; i++) {
        token_count[i].count = unsorted_token_count[i];
        token_count[i].index = i;
    }

    sort(token_count.begin(), token_count.end());
}

int main(int argc, char ** argv)
{
    if (argc != 3) {
        fprintf(stderr, "example usage: text.bin embeddings.bin\n");
        return 1;
    }

    setlocale(LC_ALL, "en_US.UTF-8");
    read_text(argv[1]);
    get_token_count();
    
    embeddings.resize(n_words);
    
    for (auto &x : embeddings) {
        for (int i = 0; i < EMBEDDING_SIZE; i++) {
            double r = 0.5 / (double)n_words;
            x.arr[i] = randrange(-1, 1);
        }
    }

    run_sliding_context();

    FILE *fout = fopen(argv[2], "wb");
    if (fout == NULL) {
        fprintf(stderr, "failed to open %s\n", argv[2]);
        return 1;
    }

    int embeddings_len = embeddings.size();
    int embedding_size = EMBEDDING_SIZE;
    
    fwrite(&embeddings_len, sizeof(int), 1, fout);
    fwrite(&embedding_size, sizeof(int), 1, fout);

    for (int i = 0; i < embeddings.size(); i++) {
        fwrite(embeddings[i].arr, sizeof(double) * EMBEDDING_SIZE, 1, fout);
    }
    fclose(fout);
}
