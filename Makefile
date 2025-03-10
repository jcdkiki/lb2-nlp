.PHONY: all clean

all: embedding

embedding: embedding.cpp
	g++ embedding.cpp -o embedding

clean:
	rm embedding
