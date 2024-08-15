+++
title = 'Python'
date = 2024-08-02T09:39:53-07:00
draft = false
+++

This Python code is the foundation for other implementations. I chose Python because:

1. TinyGrad is written in Python and my implementation closely follows an early version -- all credit to GeoHot
2. Using numpy (a tensor library without autograd) lets me focus on implementing core autograd functionality

There are two key data structures: `Function` and `Tensor`. 

# Tensor

```python
class Tensor:
    def __init__(self, buf: np.ndarray):
        self.buf = buf
        self.grad = None
        self._ctx = None
```

The Tensor class represents the core data structure in this system. Each Tensor contains:

- A `numpy` array (buf) to store its data
- A `grad` attribute for gradients
- A `_ctx` attribute to maintain a reference to the Function that created it

This _ctx is crucial for the backward pass. The backward method in Tensor implements the backpropagation algorithm. It starts from a scalar output (when implicit is True) and recursively applies the chain rule through the computational graph defined by the _ctx references.

# Function

The Function class serves as the foundation for all operations in this autograd implementation.
It defines a structure where each operation (which implements Function) has its own forward and backward methods.
```python
class Function:
    def __init__(self, *tensors : Tensor):
        self.parents = tensors
    @classmethod
    def apply(fxn: Callable[Function], *tensors: Tensor):  # pyright: ignore
        ctx = fxn(*tensors)
        ret = Tensor(ctx.forward(*[t.buf for t in tensors]))
        ret._ctx = ctx
        return ret

class Mul(Function):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x, self.y = x, y
        return x * y
    def backward(self, grad_output: np.ndarray):
        return [grad_output * self.y, grad_output * self.x]
```

The key aspect of Function is `apply`, which acts as a bridge between Tensor objects and their underlying numpy array representations.

When apply is called, the `Function` init method is instantiated with the input tensors to the operation,
storing information for backprop.
It then executes the forward pass on the raw numpy arrays (accessed via the buf attribute of the input Tensors),
but returns the result wrapped in a new Tensor with the context attached.
This allows the apply method to be used in Tensor operations while keeping the actual computations at the numpy level.

Now, with that prelude, here are all 136 lines of `tensor.py` below, along with clarifying comments.

```python
from __future__ import annotations # lets you add typing of Function inside Function
import numpy as np
from typing import Callable, List


class Function:
    def __init__(self, *tensors: Tensor):
        self.parents = tensors

    @classmethod
    def apply(fxn: Callable[Function], *tensors: Tensor):
        ctx = fxn(*tensors)
        ret = Tensor(ctx.forward(*[t.buf for t in tensors]))
        ret._ctx = ctx
        return ret


class Tensor:
    def __init__(self, buf: np.ndarray):
        self.buf = buf
        self.grad = None
        self._ctx = None

    def __repr__(self): # custom printing
        return f"Tensor with shape: {self.buf.shape}"

    # backprop
    def backward(self, implicit: bool = True):
        # base case - return nothing if we reached leaf node
        if self._ctx is None: 
            return

        # backprop always starts with an implicit gradient - dL/dL = 1.0
        if implicit:
            assert self.buf.size == 1, "Can only backprop scalar"
            self.grad = np.array(1.0)

        # current grad must already have been set from previous backward
        assert self.grad is not None

        # calculate gradients for child tensors
        grads = self._ctx.backward(self.grad)

        # apply gradients to child tensors
        for t, g in zip(self._ctx.parents, grads):
            assert (
                g.shape == t.buf.shape
            ), f"grad shape {g.shape} != tensor shape {t.buf.shape}"

            t.grad = g
            t.backward(False) # recursively call backward - but this time not implicit!

    # ops - wrappers around Function implementations

    def __mul__(self, other: Tensor) -> Tensor:
        return Mul.apply(self, other)

    def relu(self):
        return ReLU.apply(self)

    def dot(self, other: Tensor):
        return Dot.apply(self, other)

    def sum(self):
        return Sum.apply(self)

    def logsoftmax(self):
        return LogSoftmax.apply(self)

    def mean(self):
        div = Tensor(np.array([1 / self.buf.size]))
        return self.sum() * div


# anything can be saved inside forward for use inside backprop
class Mul(Function):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x, self.y = x, y
        return x * y

    def backward(self, grad_output: np.ndarray):
        return [grad_output * self.y, grad_output * self.x]


class ReLU(Function):
    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.x < 0] = 0
        return [grad_input]


class Dot(Function):
    def forward(self, x: np.ndarray, y: np.ndarray):
        self.x, self.y = x, y
        return x.dot(y)

    def backward(self, grad_output: np.ndarray):
        grad_x = grad_output.dot(self.y.T)
        grad_y = grad_output.T.dot(self.x).T
        return [grad_x, grad_y]


class Sum(Function):
    def forward(self, x: np.ndarray):
        self.x = x
        return np.array([x.sum()])

    def backward(self, grad_output: np.ndarray):
        return [np.full_like(self.x, grad_output)]


class LogSoftmax(Function):
    def forward(self, x: np.ndarray):
        def logsumexp(x):
            c = x.max(axis=1)
            return c + np.log(np.exp(x - c.reshape((-1, 1))).sum(axis=1))

        self.output = x - logsumexp(x).reshape((-1, 1))
        return self.output

    def backward(self, grad_output):
        return [
            grad_output - np.exp(self.output) * grad_output.sum(axis=1).reshape((-1, 1))
        ]

# update params and zero grad
class SGD:
    def __init__(self, params: List[Tensor], lr: float):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            p.buf -= p.grad * self.lr
            p.grad = None
```

The entire framework can be tested in a single function, just by comparing the expected output and gradients with PyTorch.
The same test is used for other implementations.

```python
import numpy as np
import torch
from tensor import Tensor

x_init = np.random.randn(128, 784).astype(np.float32)
W_init = np.random.randn(784, 128).astype(np.float32)
m_init = np.random.randn(128, 128).astype(np.float32)


def test():
    x = Tensor(x_init)
    W = Tensor(W_init)
    m = Tensor(m_init)
    out = x.dot(W)
    outr = out.relu()
    outl = outr.logsoftmax()
    outm = outl * m
    outx = outm.sum()
    outx.backward()
    return outx.buf, x.grad, W.grad


def test_pytorch():
    x = torch.tensor(x_init, requires_grad=True)
    W = torch.tensor(W_init, requires_grad=True)
    m = torch.tensor(m_init)
    out = x.matmul(W)
    outr = out.relu()
    outl = torch.nn.functional.log_softmax(outr, dim=1)
    outm = outl.mul(m)
    outx = outm.sum()
    outx.backward()
    return outx.detach().numpy(), x.grad, W.grad


for x, y in zip(test(), test_pytorch()):
    np.testing.assert_allclose(x, y, atol=1e-6)  # pyright: ignore
print("Passed!")
```

Once all tensor tests have passed, the only thing left is to implement the mnist training script.

```python
import numpy as np
from tensor import Tensor, SGD
import urllib.request
import gzip
import os
import argparse
from tqdm import trange


def fetch_mnist_download():
    def download_and_parse(url):
        with urllib.request.urlopen(url) as response:
            assert response.status == 200
            with gzip.open(response) as gz:
                return np.frombuffer(gz.read(), dtype=np.uint8).copy()

    BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    X_train = (
        download_and_parse(f"{BASE_URL}train-images-idx3-ubyte.gz")[0x10:]
        .reshape((-1, 28 * 28))
        .astype(np.float32)
    )
    Y_train = download_and_parse(f"{BASE_URL}train-labels-idx1-ubyte.gz")[8:].astype(
        np.int8
    )
    X_test = (
        download_and_parse(f"{BASE_URL}t10k-images-idx3-ubyte.gz")[0x10:]
        .reshape((-1, 28 * 28))
        .astype(np.float32)
    )
    Y_test = download_and_parse(f"{BASE_URL}t10k-labels-idx1-ubyte.gz")[8:].astype(
        np.int8
    )
    return X_train, Y_train, X_test, Y_test


def fetch_mnist_local():
    def parse_gz_file(filename):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "..", "data", filename)
        with gzip.open(file_path, "rb") as gz:
            return np.frombuffer(gz.read(), dtype=np.uint8).copy()

    X_train = (
        parse_gz_file("train-images-idx3-ubyte.gz")[0x10:]
        .reshape((-1, 28 * 28))
        .astype(np.float32)
    )
    Y_train = parse_gz_file("train-labels-idx1-ubyte.gz")[8:].astype(np.int8)
    X_test = (
        parse_gz_file("t10k-images-idx3-ubyte.gz")[0x10:]
        .reshape((-1, 28 * 28))
        .astype(np.float32)
    )
    Y_test = parse_gz_file("t10k-labels-idx1-ubyte.gz")[8:].astype(np.int8)
    return X_train, Y_train, X_test, Y_test


def layer_init(m, h):
    return (np.random.uniform(-1.0, 1.0, size=(m, h)) / np.sqrt(m * h)).astype(
        np.float32
    )


class MNISTClassifier:
    def __init__(self):
        self.l1 = Tensor(layer_init(784, 128))
        self.l2 = Tensor(layer_init(128, 10))

    def __call__(self, x):
        return x.dot(self.l1).relu().dot(self.l2).logsoftmax()


def train_and_evaluate(X_train, Y_train, X_test, Y_test):
    model = MNISTClassifier()
    optim = SGD([model.l1, model.l2], lr=0.01)
    BS = 128
    losses, accuracies = [], []

    for i in (t := trange(1000)):
        samp = np.random.randint(0, X_train.shape[0], size=(BS))
        x = Tensor(X_train[samp])
        Y = Y_train[samp]
        y = np.zeros((len(samp), 10), np.float32)
        y[range(y.shape[0]), Y] = -1.0  # negative for *N*LL loss
        y = Tensor(y)

        # network
        outs = model(x)
        # NLL loss function
        loss = (outs * y).mean()
        loss.backward()
        optim.step()

        cat = np.argmax(outs.buf, axis=1)
        accuracy = (cat == Y).mean()

        # printing
        loss = loss.buf.item()
        losses.append(loss)
        accuracies.append(accuracy)
        t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))

    # evaluate
    def numpy_eval():
        Y_test_preds_out = model(Tensor(X_test.reshape((-1, 28 * 28))))
        Y_test_preds = np.argmax(Y_test_preds_out.buf, axis=1)
        return (Y_test == Y_test_preds).mean()

    accuracy = numpy_eval()
    print("test set accuracy is %f" % accuracy)
    assert accuracy > 0.95


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST Classifier")
    parser.add_argument("--local", action="store_true", help="Use local gzip files")
    args = parser.parse_args()

    if args.local:
        print("Using local gzip files...")
        X_train, Y_train, X_test, Y_test = fetch_mnist_local()
    else:
        print("Downloading MNIST data...")
        X_train, Y_train, X_test, Y_test = fetch_mnist_download()

    train_and_evaluate(X_train, Y_train, X_test, Y_test)

```
```text
Downloading MNIST data...
loss 0.01 accuracy 0.97: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:00<00:00, 2426.95it/s]
test set accuracy is 0.965600
```

I'll add derivations for each `backward` in a future post.
