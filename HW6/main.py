import numpy

config = {
    "beta": 0.85,
    "threshold": 1e-3,
}

'''
Graph = {
    "A": ["C"],
    "B": ["A", "C", "D"],
    "C": ["B", "A"],
    "D": [],
}
'''

N = 4
M = numpy.matrix([
    [0, 1 / 3, 1 / 2, 1 / 4],
    [0, 0,     1 / 2, 1 / 4],
    [1, 1 / 3, 0,     1 / 4],
    [0, 1 / 3, 0,     1 / 4],
])

def main():
    beta = config["beta"]
    x = numpy.matrix([1 / N] * N).T
    prev_x = numpy.matrix([0] * N).T
    
    while True in (abs(prev_x - x) >= config["threshold"]):
        prev_x = x
        x = beta * M * x + (1 - beta) / N
    
    print("Final PR:\n", x.T)

if __name__ == "__main__":
    main()