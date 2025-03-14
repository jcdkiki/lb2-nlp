import struct
import numpy as np
from tensorflow.keras.layers import Dense, Input, Embedding, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from dataclasses import dataclass
from tensorflow import keras
import sys
import time

CONTEXT_SIZE = 10
BATCH_SIZE = 128
N_EPOCHS = 10

@dataclass
class Embeddings:
    n_words: int
    size: int
    arr: list[list[float]]

def read_text():
    text_bin = open("text.bin", "rb").read()
    text_len = struct.unpack_from("i", text_bin)[0]
    return list(struct.unpack_from(f"{text_len}i", text_bin, 4))

def read_embeddings() -> Embeddings:
    data = open("embeddings.bin", "rb").read()
    n_words = struct.unpack_from("i", data, 0)[0]
    size = struct.unpack_from("i", data, 4)[0]
    l = list(struct.unpack_from(f"{size*n_words}d", data, 8))
    arr = [l[i:i+size] for i in range(0, len(l), size)]
    return Embeddings(n_words, size, arr)

def data_generator(text, embeddings):
    while True:
        # Generate random starting indices for each batch
        indices = np.random.randint(CONTEXT_SIZE, len(text) - 1, size=BATCH_SIZE)
        
        X = np.zeros((BATCH_SIZE, CONTEXT_SIZE*embeddings.size), dtype=np.float32)
        y = np.zeros((BATCH_SIZE,), dtype=np.int32)
        
        for i, idx in enumerate(indices):
            x = []
            for w in text[idx-CONTEXT_SIZE:idx]:
                x += embeddings.arr[w]
            X[i] = x
            y[i] = text[idx]
        
        yield X, y

def main() -> int:    
    np.random.seed(int(time.time()))
    embeddings = read_embeddings()

    if sys.argv[1] == "train":
        text = read_text()
        
        if sys.argv[2] == "new":
            
            model = Sequential([
                Input(shape=(embeddings.size*CONTEXT_SIZE,)),
                Dense(500, activation='relu'),
                Dense(embeddings.n_words, activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',  # Critical for memory savings
                metrics=['accuracy']
            )

        elif sys.argv[2] == "+":
            model = keras.models.load_model(sys.argv[3])

        else:
            print("specify new/+")
            return 1
        
        steps_per_epoch = (len(text) - CONTEXT_SIZE) // BATCH_SIZE
        
        model.fit(
            data_generator(text, embeddings),
            steps_per_epoch=steps_per_epoch,
            epochs=N_EPOCHS,
            callbacks=[EarlyStopping(monitor='val_loss', patience=3)]
        )

        model.save("model.keras")
    
    elif sys.argv[1] == "test":
        model = keras.models.load_model(sys.argv[2])

        test_sentence = [ x.lower() for x in sys.argv[3:]]
        words = [ s[:-1] for s in open("words.txt").readlines() if len(s) != 0 ]

        test_sequence = []
        for w in test_sentence:
            test_sequence += embeddings.arr[words.index(w)]

        if len(test_sentence) < CONTEXT_SIZE:
            test_sequence = [0]*((CONTEXT_SIZE - len(test_sentence)) * embeddings.size) + test_sequence

        result = []
        log_probs = []

        for i in range(100):
            probs = model.predict(np.array(test_sequence).reshape(1, -1))[0]
            next_word = np.argmax(probs, axis=None, out=None)
            
            log_probs.append(np.log(probs[next_word]))

            result.append(next_word)
            test_sequence = test_sequence[embeddings.size:] + embeddings.arr[next_word]

        perplexity = np.exp(-np.mean(log_probs))

        print(' '.join(test_sentence))
        print(' '.join([words[i] for i in result]))
        print(f"perplexity: {perplexity}")

    else:
        print("specify train/test")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
