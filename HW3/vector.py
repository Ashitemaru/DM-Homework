from dataclasses import dataclass
from typing import Any, List
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

config = {
    "vector_path": "./100_word_vector.txt",
}

@dataclass
class WordVector:
    name: str
    vec: List[float]

vector: List[WordVector] = []

def parse_vector() -> None:
    global vector
    handle = open(config["vector_path"], mode = "r")
    raw_vector = handle.readlines()
    raw_vector = [line.split("\t") for line in raw_vector]
    vector = [
        WordVector(
            name = x[0],
            vec = list(map(float, x[1].split(" ")))
        ) for x in raw_vector
    ]
    handle.close()

def dimension_compress(method: Any, file_name: str) -> None:
    compressed = method.fit_transform([x.vec for x in vector])

    plt.figure()
    plt.scatter(compressed[: 20, 0], compressed[: 20, 1], color = "r")
    plt.scatter(compressed[20: 40, 0], compressed[20: 40, 1], color = "g")
    plt.scatter(compressed[40: 60, 0], compressed[40: 60, 1], color = "b")
    plt.scatter(compressed[60: 80, 0], compressed[60: 80, 1], color = "k")
    plt.scatter(compressed[80: 100, 0], compressed[80: 100, 1], color = "y")
    plt.grid(True)
    plt.axhline(y = 0, color = 'k')
    plt.axvline(x = 0, color = 'k')
    plt.savefig(file_name)

def main() -> None:
    parse_vector()
    dimension_compress(PCA(n_components = 2), "PCA.png")
    dimension_compress(TSNE(n_components = 2), "t-SNE.png")

if __name__ == "__main__":
    main()