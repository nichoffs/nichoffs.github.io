+++
title = "Teaching a GPT to do Modular Addition"
date = "2024-05-26T21:07:39+02:00"
authorTwitter = "" #do not include @
cover = "" #ADD GROKKING IMAGE
tags = []
keywords = ["", ""]
description = "Exploring/reproducting 'grokking' inside a toy transformer"
showFullContent = false
readingTime = true
hideComments = false
color = "" #color from the theme settings
+++

# Prerequisite Terminology

 - **Grok (verb)** - To grok means to fully and intuitively understand. When you finally grok something, it's an Aha moment where all the lingering questions you've had are suddenly resolved. For the purposes of this article, I don't need to go any deeper than that. Everybody knows that feeling.

- **Mechanistic Interpretability (noun)** - A sub-field of deep learning studying the algorithms learned by neural-net-based architectures. It was popularized (and coined?) by researchers at Anthropic in their [Transformers Circuit Thread](https://transformer-circuits.pub). 

- **Progress Measures** - Metrics that provide information about the state of training inside the model. See 1 in credits.

- **GPT** - Generatively Pre-trained Transformer - you should know how these work before reading.

# What is grokking (mechanistically)?

Researchers in the field of mechanistic interpretability have adopted the term "grokking" to refer to a unique training behavior observed in neural networks. Grokking describes the phenomenon of "delayed generalization," where a model initially overfits the training data, reaching zero training loss, before eventually developing a more general solution to the task that significantly reduces test loss.

This grokking behavior is most apparent when the model has excess capacity relative to the dataset and can trivially memorize the training examples to achieve zero train loss. To encourage the model to find a general solution, the effective size of the model is gradually reduced over the course of training through the use of substantial weight decay.

# Reproducing grokking in a toy setting

Let's set up a simple training scenario to demonstrate grokking in action, heavily inspired by the setup used in the paper [Progress Measures for Grokking via Mechanistic Interpretability](https://arxiv.org/pdf/2301.05217).
This first part of the grokking series will pretty much focus on reproducing the results from the original paper, so credit to the original authors.

We'll train a bare-bones GPT model on the task of modular addition. The transformer will take as input two integers and an equals token, and produce as output a final logit corresponding to the result of adding the integers modulo some value.

## Modular Addition

Modular addition is a fundamental arithmetic operation where you add two integers and then take the remainder when divided by a given modulus. In more formal terms, given two integers `a` and `b` and a modulus `n`, the modular addition of `a` and `b` is:

`(a + b) mod n`

For instance, if `a = 5`, `b = 3`, and `n = 4`, then:

`(5 + 3) mod 4 = 8 mod 4 = 0`

This means that when you add 5 and 3 and then divide the sum by 4, the remainder is 0.

### Dataset Creation

For our toy dataset, the input integers will range from 0 to 112, with a modulus of 113. Using 113 as the modulus, a prime number, guarantees that every possible modular sum of two integers in our range is unique and won't repeat. This property simplifies the learning problem for the model by avoiding duplicate sums.


Recall that each sample input will have three values: the first operand, the second operand, and the equals token. Since we have every possible pair from 0-112, there will be 113^2, or 12,769, total training samples. As a result, the full feature dataset will have shape `(12769, 3)` and the output will have shape `(12769, 1)`.

```python
from tinygrad import Tensor

def make_dataset(train_test_ratio=0.3, mod=113):
    ds_len = mod * mod
    # each have shape 12769=mod*mod
    # [ [0,1,2,..,mod,0,1,2,...mod] ] mod times
    a = (
        Tensor.arange(mod, dtype=dtypes.int)
        .repeat((mod, 1))
        .flatten(0, -1)
        .unsqueeze(0)
    )
    # [ [0,0,0,...,1,1,1,...,112,112,112] ]
    b = (
        Tensor.arange(mod, dtype=dtypes.int)
        .unsqueeze(-1)
        .repeat((1, mod))
        .flatten(0, -1)
        .unsqueeze(0)
    )
    # [ [113, 113, 113,...,113, 113] ]
    equals = Tensor.full((ds_len), mod).unsqueeze(0)
    sum = a + b
    products = sum.div(mod).floor() * mod
    targets = sum - products

    ds = a.cat(b, equals, dim=0).T

    indices = Tensor.randint(
        ds_len,
        low=0,
        high=ds_len,
    )

    ds_shuffled = ds[indices].cast(dtypes.float)
    targets_shuffled = (
        targets[:, indices].cast(dtypes.float).reshape(prod(targets.shape), 1)
    )

    if train_test_ratio == None:
        return ds_shuffled, targets_shuffled

    train_cutoff = int(train_test_ratio * ds_len)

    return (
        ds_shuffled[:train_cutoff],
        targets_shuffled[:train_cutoff],
        ds_shuffled[train_cutoff:],
        targets_shuffled[train_cutoff:],
    )

```

## GPT Model Breakdown

The GPT we'll use here is barebones, much simpler than the even the venerable GPT-2. [Previous work](https://arxiv.org/pdf/2301.02679) has shown grokking on modular arithmetic tasks even with basic two-layer MLPs, so our model will certainly have more than enough capacity. Using a larger and more complex model would only make analysis more challenging, which is our primary objective.

Our GPT will be one transformer block, no bias, no layernorm. Now gaze in awe at this clean implementation.

```python
class TransformerBlock:
    def __init__(self, embed_dim, head_dim, num_heads):
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.num_heads = num_heads

        self.q = Tensor.normal(embed_dim, embed_dim)
        self.k = Tensor.normal(embed_dim, embed_dim)
        self.v = Tensor.normal(embed_dim, embed_dim)

        self.head_out = Tensor.normal(num_heads * head_dim, embed_dim)

        self.ff1 = Tensor.normal(embed_dim, 4 * embed_dim)
        self.ff2 = Tensor.normal(4 * embed_dim, embed_dim)

    def attn(self, x):
        bsz = x.shape[0]
        q, k, v = [
            x.linear(proj)
            .reshape(bsz, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
            for proj in (self.q, self.k, self.v)
        ]
        return (
            q.scaled_dot_product_attention(k, v)
            .transpose(1, 2)
            .reshape(bsz, -1, self.num_heads * self.head_dim)
            .linear(self.head_out)
        )

    def mlp(self, x):
        return x.linear(self.ff1).relu().linear(self.ff2)

    def __call__(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class GPT:
    def __init__(self, num_layers, embed_dim, vocab_size, context_length, num_heads):
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_heads = num_heads

        self.tok_embed = Tensor.normal(vocab_size, embed_dim)
        self.pos_embed = Tensor.normal(context_length, embed_dim)

        self.blocks = [
            TransformerBlock(embed_dim, embed_dim // num_heads, num_heads)
            for _ in range(num_layers)
        ]

        self.out = Tensor.normal(embed_dim, vocab_size - 1)

    def __call__(self, x):
        # input shape (B,T,C)
        bsz = x.shape[0]
        pos = (
            Tensor.arange(self.context_length)
            .one_hot(self.context_length)
            .cast(dtypes.float)[: x.shape[1]]
            .expand((bsz, None, None))
        )
        x = x.one_hot(self.vocab_size).linear(self.tok_embed) + pos.linear(
            self.pos_embed
        )
        x = x.sequential(self.blocks)
        x = x.reshape(-1, x.shape[-1]).linear(self.out)
        return x.reshape((bsz, -1, x.shape[-1]))
```

## Training

```python
def loss_fn(logits: Tensor, labels):
    log_probs = logits.log_softmax(axis=-1).cast(dtypes.float64)
    correct = log_probs.gather(idx=labels, dim=-1)[:, 0]
    return -correct.mean()

def train(
    model,
    X_train,
    Y_train,
    X_test,
    Y_test,
    optim,
    steps=10000,  # Adjust this as per the actual training epochs needed
    lossfn=lambda out, y: out.sparse_categorical_crossentropy(y),
    allow_jit=True,
):
    def train_step(x, y):
        out = model(x)[:, -1]
        loss = lossfn(out, y)
        loss.backward()
        optim.step()
        optim.zero_grad()
        return loss.realize()

    def test_step(x, y):
        out = model(x)[:, -1]
        optim.zero_grad()
        loss = lossfn(out, y)
        return loss.realize()

    if allow_jit:
        train_step = TinyJit(train_step)

    train_losses = []
    test_losses = []
    with Tensor.train():
        for i in (t := trange(steps)):
            train_loss = train_step(X_train, Y_train)
            test_loss = test_step(X_test, Y_test)

            train_losses.append(train_loss.numpy())
            test_losses.append(test_loss.numpy())

            t.set_description(
                f"train loss: {train_loss.numpy():.2f}, test loss: {test_loss.numpy():.2f}"
            )
    return train_losses, test_losses
```

## Execution - will it grok?!?

Here, we instantiate all the necessary constants defined in the original paper. There's a couple important things to notice here:

 - Epochs: 55000 - When analyzing the trained model, it's helpful to train for extra epochs so that the learned weights are as refined and stable as possible. This will make more sense once you learn the model's algorithm.

 - Weight Decay: 1 - As mentioned earlier, strong weight decay provides the pressure needed for the model to generalize rather than just memorizing the training set. Without it, there would be no incentive to find a more general solution once zero training loss is reached.

 - Training loss: .3 - This 30% train / 70% test split was found to be the smallest training set size that still reliably leads to grokking. The authors noted that "using ≥ 60% data leads to immediate generalization, while using 10% or 20% of the data doesn’t lead to grokking even after 40k epochs."

Now let's get to training.

```python
mod = 113
num_layers = 1
embed_dim = 128
vocab_size = mod
context_length = 3
num_heads = 4
num_epochs = 55000
learning_rate = 1e-3
wd = 1.0
train_test_ratio = 0.3

x_train, y_train, x_test, y_test = make_dataset(train_test_ratio, mod)

model = GPT(num_layers, embed_dim, vocab_size, context_length, num_heads)

optimizer = AdamW(get_parameters(model), lr=learning_rate, b1=0.9, b2=0.98, wd=wd)

train_losses, test_losses = train(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        optimizer,
        steps=num_epochs,
        lossfn=loss_fn,
)
```

# Analysis

Great! The model has been trained. If we see the characteristic grokking curve, we can be fairly confident that 

```python
plt.figure(figsize=(10, 5))
train_losses_log = np.log(
    np.maximum(train_losses, 1e-10)
)  # Constant to avoid log(0)
test_losses_log = np.log(np.maximum(test_losses, 1e-10))
plt.plot(train_losses_log, label="Log Training Loss")
plt.plot(test_losses_log, label="Log Testing Loss")
plt.title("Logarithm of Training and Testing Losses Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.legend()
plt.grid(True)
plt.show()
```

Here is the famous graph that I reproduced on my ipad very poorly. 
The exact training dynamics may be different than yours, so just know that the memorization (train loss=0) will occur
after a couple hundred epochs, and the beginning of clean-up(when test loss starts to drop significantly) happens around 30,000.

![Grokking Graph](/cropped.jpeg)

You'll also notice I marked off three sections: memorization, circuit formation, and clean-up. 
In part 2 of this series, we'll perform a deep dive analysis of the algorithm learned by the model.

### Credits
- 1 https://imtiazhumayun.github.io/grokking/
- 2 https://arxiv.org/pdf/2301.05217
