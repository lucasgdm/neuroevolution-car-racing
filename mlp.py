import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # sigmoid

def relu(x):
    return np.maximum(0, x)

class MyMLP:
    def __init__(self, layer_sizes, activation='relu'):
        assert len(layer_sizes) >= 2
        assert activation in ['sigmoid', 'relu']
        self.activation = sigmoid if activation == 'sigmoid' else relu
        self.ws = []
        for a, b in zip(layer_sizes[:-1], layer_sizes[1:]):
            mat = np.zeros((a + 1, b + 1))
            mat[-1, -1] = np.infty if activation == 'sigmoid' else 1
            mat[:, :-1] = np.random.rand(*mat[:, :b].shape) * 0.2 - 0.1
            self.ws.append(mat)

    def feed(self, x):
        x = np.append(x, 1)
        for w in self.ws:
            x = self.activation(x @ w)
            assert x[-1] == 1
        # Neural net output is [steer right, gas, brake, steer left, 1], but the
        # environment expects [steer, gas, brake]
        x[0] -= x[-2]
        return x[:-2]

    def mutate(self, mutation_rate, sigma):
        for i, mat in enumerate(self.ws):
            p = np.clip(mutation_rate / np.prod(mat[:, :-1].shape), 0, 1)
            self.ws[i][:, :-1] += np.random.choice([0, 1], size=mat[:, :-1].shape, p=[1-p, p]) *\
                                  np.random.normal(*mat[:, :-1].shape) *\
                                  sigma


