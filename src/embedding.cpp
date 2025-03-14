#include <algorithm>
#include <vector>
#include <unordered_set>
#include <random>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>

static constexpr int CONTEXT_RADIUS = 5;
static constexpr int EMBEDDING_SIZE = 100;
static constexpr double LEARNING_RATE = 0.025;
static constexpr double NEG_FACTOR = 2;
static constexpr int CONTEXT_SIZE = CONTEXT_RADIUS * 2 + 1;

struct Embedding {
    double arr[EMBEDDING_SIZE];
};

struct TokenCount {
    int index;
    int count;
    bool operator<(const TokenCount& other) const {
        return count < other.count || (count == other.count && index < other.index);
    }
};

int n_words;
std::vector<int> unsorted_token_count;
std::vector<TokenCount> token_count;
std::vector<Embedding> embeddings;
std::vector<int> text;
std::mt19937 rng;

double dot(const Embedding &a, const Embedding &b) {
    double res = 0.0;
    for (int i = 0; i < EMBEDDING_SIZE; ++i) {
        res += a.arr[i] * b.arr[i];
    }
    return res;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

void find_negative_context(int mid_pos, std::vector<int>& res) {
    res.clear();
    if (mid_pos < 0 || mid_pos >= text.size()) return;
    
    int mid_word = text[mid_pos];
    std::unordered_set<int> positive_words;
    int start = std::max(0, mid_pos - CONTEXT_RADIUS);
    int end = std::min((int)text.size(), mid_pos + CONTEXT_RADIUS + 1);
    for (int i = start; i < end; ++i) {
        positive_words.insert(text[i]);
    }

    int word_count = unsorted_token_count[mid_word];
    int count_range = std::max(1, word_count / 10);
    auto left = std::lower_bound(token_count.begin(), token_count.end(), TokenCount{0, word_count - count_range});
    auto right = std::upper_bound(token_count.begin(), token_count.end(), TokenCount{0, word_count + count_range});

    std::vector<int> candidates;
    while (candidates.size() < CONTEXT_RADIUS * NEG_FACTOR * 2 && left != right) {
        int idx = left->index;
        if (!positive_words.count(idx) && idx != mid_word) {
            candidates.push_back(idx);
        }
        ++left;
    }

    std::shuffle(candidates.begin(), candidates.end(), rng);
    for (int i = 0; i < candidates.size() && res.size() < CONTEXT_RADIUS * NEG_FACTOR; ++i) {
        res.push_back(candidates[i]);
    }

    // not enough words
    std::uniform_int_distribution<int> dist(0, n_words - 1);
    while (res.size() < CONTEXT_RADIUS * NEG_FACTOR) {
        int word = dist(rng);
        if (!positive_words.count(word) && word != mid_word) {
            res.push_back(word);
        }
    }
}

void run_sliding_context() {
    std::vector<int> negative_context;
    
    for (int left = 0; left <= text.size() - CONTEXT_SIZE; ++left) {
        int mid_pos = left + CONTEXT_RADIUS;
        int mid_word = text[mid_pos];
        Embedding& center_emb = embeddings[mid_word];

        for (int i = left; i < left + CONTEXT_SIZE; ++i) {
            if (i == mid_pos) continue;
            int context_word = text[i];
            Embedding& context_emb = embeddings[context_word];
            double score = dot(center_emb, context_emb);
            double gradient = (sigmoid(score) - 1.0) * LEARNING_RATE;

            for (int j = 0; j < EMBEDDING_SIZE; ++j) {
                double update = gradient * context_emb.arr[j];
                center_emb.arr[j] -= update;
                context_emb.arr[j] -= gradient * center_emb.arr[j];
            }
        }

        find_negative_context(mid_pos, negative_context);
        for (int neg_word : negative_context) {
            Embedding& neg_emb = embeddings[neg_word];
            double score = dot(center_emb, neg_emb);
            double gradient = sigmoid(score) * LEARNING_RATE;

            for (int j = 0; j < EMBEDDING_SIZE; ++j) {
                double update = gradient * neg_emb.arr[j];
                center_emb.arr[j] -= update;
                neg_emb.arr[j] -= gradient * center_emb.arr[j];
            }
        }
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
