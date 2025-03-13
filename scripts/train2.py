import struct
import numpy as np
from tensorflow import keras
import tensorflow as tf
from dataclasses import dataclass
import sys

CONTEXT_SIZE = 6
BATCH_SIZE = 16
N_EPOCHS = 5

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
        indices = np.random.randint(CONTEXT_SIZE, len(text) - 1, size=BATCH_SIZE)
        
        X = np.zeros((BATCH_SIZE, CONTEXT_SIZE*embeddings.size), dtype=np.float64)
        y = np.zeros((BATCH_SIZE, embeddings.size), dtype=np.float64)
        
        for i, idx in enumerate(indices):
            x = []
            for w in text[idx-CONTEXT_SIZE:idx]:
                x += embeddings.arr[w]
            X[i] = np.array(x, dtype=np.float64)
            y[i] = embeddings.arr[text[idx]]

        yield X, y

def get_closest_word(embeddings, target : np.array):
    #return min(range(embeddings.n_words), key = lambda x: np.linalg.norm(np.array(embeddings.arr[x]) - target))
    target_norm = np.linalg.norm(target)
    return min(range(embeddings.n_words), key = lambda x: 1 - np.dot(embeddings.arr[x], target) / np.linalg.norm(embeddings.arr[x]) / target_norm)


def main() -> int:
    if len(sys.argv) == 1:
        print("specify train/test")
        return 1

    embeddings = read_embeddings()

    model = keras.Sequential([
        keras.layers.Input(shape=(embeddings.size * CONTEXT_SIZE,), dtype="float64"),
        keras.layers.Dense(500, activation='relu', dtype="float64"),
        keras.layers.Dense(embeddings.size, dtype="float64")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=keras.losses.MeanSquaredError(),
    )

    if sys.argv[1] == "train":
        text = read_text()
        
        model.fit(
            data_generator(text, embeddings),
            steps_per_epoch=10000,
            epochs=N_EPOCHS,
        )

        model.save_weights("model.weights.h5")
    
    elif sys.argv[1] == "test":
        model.build(input_shape=(None, CONTEXT_SIZE * embeddings.size))
        model.load_weights("model.weights.h5")

        test_sentence = "в бою умер князь андрей болконский".split()
        words = [ s[:-1] for s in open("words.txt").readlines() if len(s) != 0 ]

        test_sequence = []
        for w in test_sentence:
            test_sequence += embeddings.arr[words.index(w)]

        dot = words.index('.')
        if len(test_sequence) < CONTEXT_SIZE:
            test_sequence = embeddings.arr[dot]*(CONTEXT_SIZE - len(test_sequence)) + test_sequence

        result = []

        for i in range(5):
            next_embedding = model.predict(np.array([test_sequence]))[0]
            next_word = get_closest_word(embeddings, next_embedding)
            
            result.append(next_word)
            test_sequence = test_sequence[embeddings.size:] + embeddings.arr[next_word]

        print("result:")
        print(f"{' '.join(test_sentence)}..........")
        print(' '.join([words[i] for i in result]))

    else:
        print("specify train/test")
        return 1
    
    return 0

if __name__ == "__main__":
    # Hide GPU from visible devices
    #tf.config.set_visible_devices([], 'GPU')
    exit(main())
