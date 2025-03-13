#ifndef LB2_SPLIT_TEXT_COMMON_H
#define LB2_SPLIT_TEXT_COMMON_H

#include <vector>

void write_words(const char *filename, const std::vector<const wchar_t*> &words);
void write_text(const char *filename, const std::vector<int> &text);

#endif
