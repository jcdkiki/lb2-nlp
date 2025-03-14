from bpe import BPETokenizerSimple
import sys
import struct 

text = open(sys.argv[1]).read()

bpe = BPETokenizerSimple()
bpe.load_vocab_and_merges("vocab", "merges")

words = []
for i in range(15000):
    words.append(bpe.vocab[i])

words_file = open("words.txt", "w")
words_file.write('\n'.join(words))
words_file.close()

encoded = bpe.encode(text)

b = struct.pack(f"i {len(encoded)}i", len(encoded), *encoded)

text_file = open("text.bin", "wb")
text_file.write(b)
text_file.close()
