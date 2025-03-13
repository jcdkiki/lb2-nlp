import struct
import numpy as np
from tensorflow.keras.layers import Dense, Input, Embedding, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from dataclasses import dataclass
import sys

CONTEXT_SIZE = 6
BATCH_SIZE = 256

@dataclass
class Embeddings:
    n_words: int
    size: int
    arr: np.ndarray

def read_text():
    text_bin = open("text.bin", "rb").read()
    text_len = struct.unpack_from("i", text_bin)[0]
    return list(struct.unpack_from(f"{text_len}i", text_bin, 4))

def read_embeddings() -> Embeddings:
    data = open("embeddings.bin", "rb").read()
    n_words = struct.unpack_from("i", data, 0)[0]
    size = struct.unpack_from("i", data, 4)[0]
    l = list(struct.unpack_from(f"{size*n_words}d", data, 8))
    arr = np.array([l[i:i+size] for i in range(0, len(l), size)], dtype=np.float32)
    return Embeddings(n_words, size, arr)

def data_generator(text, context_size, batch_size):
    while True:
        # Generate random starting indices for each batch
        indices = np.random.randint(context_size, len(text) - 1, size=batch_size)
        
        X = np.zeros((batch_size, context_size), dtype=np.int32)
        y = np.zeros((batch_size,), dtype=np.int32)
        
        for i, idx in enumerate(indices):
            X[i] = text[idx-context_size:idx]
            y[i] = text[idx]
        
        yield X, y

def main() -> int:    
    text = read_text()
    embeddings = read_embeddings()
    
    model = Sequential([
        Embedding(
            input_dim=embeddings.n_words,
            output_dim=embeddings.size,
            weights=[embeddings.arr],
            input_length=CONTEXT_SIZE,
            trainable=False
        ),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(embeddings.n_words, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    if sys.argv[1] == "train":
        # Calculate steps per epoch based on dataset size
        steps_per_epoch = (len(text) - CONTEXT_SIZE) // BATCH_SIZE
        
        model.fit(
            data_generator(text, CONTEXT_SIZE, BATCH_SIZE),
            steps_per_epoch=steps_per_epoch,
            epochs=10
        )

        model.save_weights("model.weights.h5")
    
    elif sys.argv[1] == "test":
        model.build(input_shape=(None, CONTEXT_SIZE))
        model.load_weights("model.weights.h5")

        test_sentence = "в бою умер князь андрей болконский".split()
        #test_sentence = input("input text: ").split()
        words = [ s[:-1] for s in open("words.txt").readlines() if len(s) != 0 ]


        test_sequence = []
        for w in test_sentence:
            test_sequence.append(words.index(w))

        if len(test_sequence) < CONTEXT_SIZE:
            test_sequence = [0]*(CONTEXT_SIZE - len(test_sequence)) + test_sequence

        result = []

        for i in range(20):
            print(test_sequence)
            probs = model.predict(np.array([test_sequence]), verbose=0)[0]
            next_word = np.argmax(probs, axis=None, out=None)
            result.append(next_word)
            test_sequence = test_sequence[1:] + [ int(next_word) ]

        print(' '.join(test_sentence))
        print(' '.join([words[i] for i in result]))

    else:
        print("specify train/test")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
