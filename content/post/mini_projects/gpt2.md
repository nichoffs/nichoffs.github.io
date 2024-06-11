+++
title = 'TinyGrad GPT2 à la Karpathy - Part 1'
date = 2024-06-11T16:25:35+02:00
draft = false
+++

Whenever Andrej Karpathy releases a new [YouTube video](https://www.youtube.com/watch?v=l8pRSuU81PU&t=9565s), the only moral thing to do is drop all responsibilites and replicate it in TinyGrad. This released yesterday (June 10) and is over 4 hours long (I'm salivating), so I've only finished the first part. I anticipate the later parts of the video will include more Torch-specific optimizations, which will make things a bit more difficult.

# Setting up inference with pre-trained weights

GPT2 has four models from 124M to 1.5B.

| Model       | Number of Layers | Number of Heads | Embedding Size | Parameter Count |
|-------------|------------------|-----------------|----------------|-----------------|
| GPT-2 Small | 12               | 12              | 768            | ~124M           |
| GPT-2 Medium| 24               | 16              | 1024           | ~350M           |
| GPT-2 Large | 36               | 20              | 1280           | ~774M           |
| GPT-2 XL    | 48               | 25              | 1600           | ~1.5B           |

We'll primarily focus on the 124M model, but the parameter loading method named `build` that we'll be designing here can work with any.

First, let's outline the configuration of each model size, matching the naming conventions used by the `transformers` library.

```python
@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    norm_eps: float = 1e-5

@dataclass
class GPT2Small(GPT2Config):
    pass

@dataclass
class GPT2Medium(GPT2Config):
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1024

@dataclass
class GPT2Large(GPT2Config):
    n_layer: int = 36
    n_head: int = 20
    n_embd: int = 1280

@dataclass
class GPT2XL(GPT2Config):
    n_layer: int = 48
    n_head: int = 25
    n_embd: int = 1600
    
MODEL_CONFIGS = {
    'gpt2': GPT2Small,
    'gpt2-medium': GPT2Medium,
    'gpt2-large': GPT2Large,
    'gpt2-xl': GPT2XL
}
```


To load the pre-trained GPT2, the names of the parameters must match the attributes of our classes. Let's see what these names are.

```python
weights = torch_load(fetch(f'https://huggingface.co/gpt2/resolve/main/pytorch_model.bin'))
get_state_dict(weights).keys()
```

```text
dict_keys(['wte.weight', 'wpe.weight', 'h.0.ln_1.weight', 'h.0.ln_1.bias', 'h.0.attn.bias', 'h.0.attn.c_attn.weight', 'h.0.attn.c_attn.bias', 'h.0.attn.c_proj.weight', 'h.0.attn.c_proj.bias', 'h.0.ln_2.weight', 'h.0.ln_2.bias', 'h.0.mlp.c_fc.weight', 'h.0.mlp.c_fc.bias', 'h.0.mlp.c_proj.weight', 'h.0.mlp.c_proj.bias', 'h.1.ln_1.weight', 'h.1.ln_1.bias', 'h.1.attn.bias', 'h.1.attn.c_attn.weight', 'h.1.attn.c_attn.bias', 'h.1.attn.c_proj.weight', 'h.1.attn.c_proj.bias', 'h.1.ln_2.weight', 'h.1.ln_2.bias', 'h.1.mlp.c_fc.weight', 'h.1.mlp.c_fc.bias', 'h.1.mlp.c_proj.weight', 'h.1.mlp.c_proj.bias', 'h.2.ln_1.weight', 'h.2.ln_1.bias', 'h.2.attn.bias', 'h.2.attn.c_attn.weight', 'h.2.attn.c_attn.bias', 'h.2.attn.c_proj.weight', 'h.2.attn.c_proj.bias', 'h.2.ln_2.weight', 'h.2.ln_2.bias', 'h.2.mlp.c_fc.weight', 'h.2.mlp.c_fc.bias', 'h.2.mlp.c_proj.weight', 'h.2.mlp.c_proj.bias', 'h.3.ln_1.weight', 'h.3.ln_1.bias', 'h.3.attn.bias', 'h.3.attn.c_attn.weight', 'h.3.attn.c_attn.bias', 'h.3.attn.c_proj.weight', 'h.3.attn.c_proj.bias', 'h.3.ln_2.weight', 'h.3.ln_2.bias', 'h.3.mlp.c_fc.weight', 'h.3.mlp.c_fc.bias', 'h.3.mlp.c_proj.weight', 'h.3.mlp.c_proj.bias', 'h.4.ln_1.weight', 'h.4.ln_1.bias', 'h.4.attn.bias', 'h.4.attn.c_attn.weight', 'h.4.attn.c_attn.bias', 'h.4.attn.c_proj.weight', 'h.4.attn.c_proj.bias', 'h.4.ln_2.weight', 'h.4.ln_2.bias', 'h.4.mlp.c_fc.weight', 'h.4.mlp.c_fc.bias', 'h.4.mlp.c_proj.weight', 'h.4.mlp.c_proj.bias', 'h.5.ln_1.weight', 'h.5.ln_1.bias', 'h.5.attn.bias', 'h.5.attn.c_attn.weight', 'h.5.attn.c_attn.bias', 'h.5.attn.c_proj.weight', 'h.5.attn.c_proj.bias', 'h.5.ln_2.weight', 'h.5.ln_2.bias', 'h.5.mlp.c_fc.weight', 'h.5.mlp.c_fc.bias', 'h.5.mlp.c_proj.weight', 'h.5.mlp.c_proj.bias', 'h.6.ln_1.weight', 'h.6.ln_1.bias', 'h.6.attn.bias', 'h.6.attn.c_attn.weight', 'h.6.attn.c_attn.bias', 'h.6.attn.c_proj.weight', 'h.6.attn.c_proj.bias', 'h.6.ln_2.weight', 'h.6.ln_2.bias', 'h.6.mlp.c_fc.weight', 'h.6.mlp.c_fc.bias', 'h.6.mlp.c_proj.weight', 'h.6.mlp.c_proj.bias', 'h.7.ln_1.weight', 'h.7.ln_1.bias', 'h.7.attn.bias', 'h.7.attn.c_attn.weight', 'h.7.attn.c_attn.bias', 'h.7.attn.c_proj.weight', 'h.7.attn.c_proj.bias', 'h.7.ln_2.weight', 'h.7.ln_2.bias', 'h.7.mlp.c_fc.weight', 'h.7.mlp.c_fc.bias', 'h.7.mlp.c_proj.weight', 'h.7.mlp.c_proj.bias', 'h.8.ln_1.weight', 'h.8.ln_1.bias', 'h.8.attn.bias', 'h.8.attn.c_attn.weight', 'h.8.attn.c_attn.bias', 'h.8.attn.c_proj.weight', 'h.8.attn.c_proj.bias', 'h.8.ln_2.weight', 'h.8.ln_2.bias', 'h.8.mlp.c_fc.weight', 'h.8.mlp.c_fc.bias', 'h.8.mlp.c_proj.weight', 'h.8.mlp.c_proj.bias', 'h.9.ln_1.weight', 'h.9.ln_1.bias', 'h.9.attn.bias', 'h.9.attn.c_attn.weight', 'h.9.attn.c_attn.bias', 'h.9.attn.c_proj.weight', 'h.9.attn.c_proj.bias', 'h.9.ln_2.weight', 'h.9.ln_2.bias', 'h.9.mlp.c_fc.weight', 'h.9.mlp.c_fc.bias', 'h.9.mlp.c_proj.weight', 'h.9.mlp.c_proj.bias', 'h.10.ln_1.weight', 'h.10.ln_1.bias', 'h.10.attn.bias', 'h.10.attn.c_attn.weight', 'h.10.attn.c_attn.bias', 'h.10.attn.c_proj.weight', 'h.10.attn.c_proj.bias', 'h.10.ln_2.weight', 'h.10.ln_2.bias', 'h.10.mlp.c_fc.weight', 'h.10.mlp.c_fc.bias', 'h.10.mlp.c_proj.weight', 'h.10.mlp.c_proj.bias', 'h.11.ln_1.weight', 'h.11.ln_1.bias', 'h.11.attn.bias', 'h.11.attn.c_attn.weight', 'h.11.attn.c_attn.bias', 'h.11.attn.c_proj.weight', 'h.11.attn.c_proj.bias', 'h.11.ln_2.weight', 'h.11.ln_2.bias', 'h.11.mlp.c_fc.weight', 'h.11.mlp.c_fc.bias', 'h.11.mlp.c_proj.weight', 'h.11.mlp.c_proj.bias', 'ln_f.weight', 'ln_f.bias'])
```

Here's a concise summary of the naming conventions. When I say `weight+bias`, that means that this component will have a `.weight` and `.bias` tensor component.

### `Transformer` Parameters:
* Token Embedding(weight only) = `wte`
* Positional Embedding(weight only) = `wpe`
* Transformer Block(see below) at layer `n` = `h.n`
* Final Layer-Norm(weight+bias): `ln_f`

Notice there is no logit head at the end. GPT2 shares weights for the token embedding and output head. We'll explicitly tie these later when constructing the classes.

You may have noticed there's a `bias` parameter for each attention block representing the attention mask. We don't need this because TinyGrad's `scaled_dot_product_attention` method automatically handles masking. TinyGrad will automatically ignore these when loading weights. 

### `TransformerBlock` Parameters:
* Attention Pre-Norm(weight+bias): `ln_1`
* Attention(see below): `attn`
* MLP Pre-Norm(weight+bias): `mlp`
* MLP(see below): `mlp`

### `Attention` Parameters:
* QKV projection(weight+bias): `c_attn`
* Output Projection(weight+bias): `c_proj`

### `MLP` Parameters:
* Expansion Projection(weight+bias): `c_fc`
* Compression Projection(weight+bias): `c_proj`

Now let's construct the shell of each class as well as the final build method. Notice the tying of `lm_head` and `wte` in both the `build` and initialization method.

```python
class Attention:
    def __init__(self, config : GPT2Config):
        self.c_attn = Linear(config.n_embd, config.n_embd*3)
        self.c_proj = Linear(config.n_embd, config.n_embd)

class MLP:
    def __init__(self, config : GPT2Config):
        self.c_fc = Linear(config.n_embd, config.n_embd*4)
        self.c_proj = Linear(config.n_embd*4, config.n_embd)

class TransformerBlock:
    def __init__(self, config : GPT2Config):
        self.ln_1 = LayerNorm(config.n_embd, eps=config.norm_eps)
        self.ln_2 = LayerNorm(config.n_embd, eps=config.norm_eps)
        self.attn = Attention(config)
        self.mlp = MLP(config)

class GPT2:
    def __init__(self, config : GPT2Config):
        self.config = config

        # self.decoder = []
        self.wte = Embedding(config.vocab_size, config.n_embd)
        self.wpe = Embedding(config.block_size, config.n_embd)
        self.h = [TransformerBlock(config) for _ in range(config.n_layer)]
        self.ln_f = LayerNorm(config.n_embd, eps=config.norm_eps)
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)

        self.wte.weight = self.lm_head.weight

    @staticmethod
    def build(MODEL_NAME):
        
        weights = torch_load(fetch(f'https://huggingface.co/{MODEL_NAME}/resolve/main/pytorch_model.bin'))
        weights['lm_head.weight'] = weights['wte.weight']

        # I believe this is necessary because TinyGrad linear matmul acts on the other side
        transposed = ('attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight')
        for k in weights:
            if k.endswith(transposed):
                weights[k] = weights[k].T

        return load_state_dict(GPT2(MODEL_CONFIGS[MODEL_NAME]), weights)
```

If everything has gone according to plan, we should be able to load the model without error.

```python
GPT2.build('gpt2')
```
```text
ram used:  0.50 GB, lm_head.weight: 100%|██████████| 149/149 [00:00<00:00, 588.23it/s]
loaded weights in 261.49 ms, 0.65 GB loaded at 2.49 GB/s
```

Great. I won't show it here, but I've verified that the parameters are equivalent to those directly using `transformers`. It's time to begin implementing the forward-pass logic for each head, starting with `MLP`.

```python
class MLP:
    def __init__(self, config : GPT2Config):
        self.c_fc = Linear(config.n_embd, config.n_embd*4)
        self.c_proj = Linear(config.n_embd*4, config.n_embd)
    def __call__(self, x):
        x = self.c_fc(x).gelu()
        x = self.c_proj(x)
        return x
```

Next is `Attention`, Multi-Headed Attention that is. I "cleaned up" some of the code to my liking, but it behaves the exact same and is essentially a direct equivalence.

```python
class Attention:
    def __init__(self, config: GPT2Config):
        self.config = config
        self.c_attn = Linear(config.n_embd, config.n_embd*3)
        self.c_proj = Linear(config.n_embd, config.n_embd)
    def __call__(self, x):
        B,T,C = x.shape

        q, k, v = self.c_attn(x).split(C, dim=-1) #(B,T,3C) -> (B,T,C) x 3
        split_heads = lambda x: x.view(B, T, self.config.n_head, self.config.n_embd//self.config.n_head).transpose(1,2)
        q, k, v = map(split_heads, (q,k,v))

        y = q.scaled_dot_product_attention(k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y
```

Lastly, the great `TransformerBlock`.

```python
class TransformerBlock:
    def __init__(self, config : GPT2Config):
        self.ln_1 = LayerNorm(config.n_embd, eps=config.norm_eps)
        self.ln_2 = LayerNorm(config.n_embd, eps=config.norm_eps)
        self.attn = Attention(config)
        self.mlp = MLP(config)
    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

Each component's forward pass is built, so we can finally complete (at least for slow inference) the `GPT2` class implementation.

```python
class GPT2:
    def __init__(self, config : GPT2Config = GPT2Small):
        self.config = config

        self.wte = Embedding(config.vocab_size, config.n_embd)
        self.wpe = Embedding(config.block_size, config.n_embd)
        self.h = [TransformerBlock(config) for _ in range(config.n_layer)]
        self.ln_f = LayerNorm(config.n_embd, config.norm_eps)
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)

        # tie weights - HUGE SAVINGS
        self.lm_head.weight = self.wte.weight
    
    
    def __call__(self, idx):
        B,T = idx.shape

        assert T <= self.config.block_size, f"Cannot forward, model block size is {self.config.block_size} but got sequence of length {T}"
        pos = Tensor.arange(0, T, dtype=dtypes.long) # (T,)
        pos_emb = self.wpe(pos) # (T,) -> (T,C)
        tok_emb = self.wte(idx) # (B,T) -> (B,T,C)

        x = tok_emb + pos_emb
        x = x.sequential(self.h)

        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,C) -> (B,T,V)

        return logits

    @staticmethod
    def build(MODEL_NAME):
        
        weights = torch_load(fetch(f'https://huggingface.co/{MODEL_NAME}/resolve/main/pytorch_model.bin'))

        transposed = ('attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight')
        for k in weights:
            if k.endswith(transposed):
                weights[k] = weights[k].T

        weights['lm_head.weight'] = weights['wte.weight']
        model = GPT2(MODEL_CONFIGS[MODEL_NAME])
        load_state_dict(model, weights)

        return model
```

Cool. Let's see if we can load a pre-trained model and run inference on it. TinyGrad doesn't have a `Tensor.topk` implementation, so I had to borrow from [this](https://github.com/tinygrad/tinygrad/blob/97b05f567e8e42a2475f8a063fb080b200f6f033/extra/models/mask_rcnn.py) model in their repo's `extra/models` folder.

Additionally, we'll use `tiktoken` for tokenizing.

```python
num_return_sequences = 5
max_length = 30

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("What is the meaning of life?")
x = Tensor(tokens, dtype=dtypes.long).unsqueeze(0).repeat(num_return_sequences, 1)

model = GPT2.build('gpt2-large')

#does this do anything?
Tensor.no_grad = True
Tensor.training = False
while x.shape[1] < max_length:
    logits = model(x)
    logits = logits[:, -1, :]
    probs = logits.softmax(-1)
    topk_probs, topk_indices = topk(probs, 50, dim=-1)
    ix = topk_probs.multinomial(1)
    xcol = topk_indices.gather(-1, ix)
    x = x.cat(xcol, dim=1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].numpy().tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
```

```text
> What is the meaning of life? What is our purpose in this world? Is there another meaning?" In terms of the meaning of life, the answer
> What is the meaning of life? What is God? What is eternity? When did Adam first become alive? Does God exist? Is God a being
> What is the meaning of life? How does the life of a human being and all of its consequences and results mean to us? How can it be
> What is the meaning of life?

Well, how in the world are you going to survive?

What is life?

Well
> What is the meaning of life?

Some people may claim that there is nothing to live for but a mere struggle for existence, a contest for
```

The new lines are annoying, but still looks good to me. Definitely better than random initialization:
```python
> Hello, I'm a language model, Yellowstone naughtyagic problemsStrong Intakeleaf quantify� Intake Debate Winchester Frem wrestling stations sufficiently 裏覚醒 drying Tut Tut stations practitioners
```

# Training

In the final part of this post, we'll get some basic training logistics sorted out.  Instead of including the loss calculation in some external training method, we can directly adapt the forward pass to accept labels and return the loss (as well as logits like we've already done).

`logits` will have shape `(B,T,V)` and targets will be `(B,T)`. `sparse_categorical_crossentropy` requires the input to be `(N, num_classes)`, where `num_classes` is `V` in our case. Here's the new `GPT2` class with the loss logic.

```python
class GPT2:
    def __init__(self, config : GPT2Config = GPT2Small):
        self.config = config

        self.wte = Embedding(config.vocab_size, config.n_embd)
        self.wpe = Embedding(config.block_size, config.n_embd)
        self.h = [TransformerBlock(config) for _ in range(config.n_layer)]
        self.ln_f = LayerNorm(config.n_embd, config.norm_eps)
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)

        # tie weights - HUGE SAVINGS
        self.lm_head.weight = self.wte.weight
    
    def __call__(self, idx, targets=None):
        B,T = idx.shape

        assert T <= self.config.block_size, f"Cannot forward, model block size is {self.config.block_size} but got sequence of length {T}"
        pos = Tensor.arange(0, T, dtype=dtypes.long) # (T,)
        pos_emb = self.wpe(pos) # (T,) -> (T,C)
        tok_emb = self.wte(idx) # (B,T) -> (B,T,C)

        x = tok_emb + pos_emb
        x = x.sequential(self.h)

        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,C) -> (B,T,V)

        if targets is not None:
            loss = logits.flatten(0,1).sparse_categorical_crossentropy(targets.flatten())
            return logits, loss.realize()

        return logits, None

    @staticmethod
    def build(MODEL_NAME):
        
        weights = torch_load(fetch(f'https://huggingface.co/{MODEL_NAME}/resolve/main/pytorch_model.bin'))

        transposed = ('attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight')
        for k in weights:
            if k.endswith(transposed):
                weights[k] = weights[k].T

        weights['lm_head.weight'] = weights['wte.weight']
        model = GPT2(MODEL_CONFIGS[MODEL_NAME])
        load_state_dict(model, weights)

        return model
```

We'll load up the tiny-shakespeare dataset in classic Karpathy style.

```python
with open("datasets/shake.txt", "r") as f:
    text = f.read()
data = text[:1000]
encoded_data = enc.encode(data)
print(data[:100])
```

```text
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You
```

Obviously, we'll need a batched input `(B,T)`, which can be accomplished using `view`. `reshape` would do the same.

For every batch `B` of text with length `T`, we need to retrieve `B*T+1`. I think this code should be self-explanatory, assuming you've gotten everything up until this point.

Before creating a `DataLoaderLite`, we should make sure we can overfit on a single batch.

```python
buf = Tensor(encoded_data[:24+1])
batch = lambda x: x.view(4,6)
x = batch(buf[:-1])
y = batch(buf[1:])

Tensor.training = True
Tensor.no_grad = False
model = GPT2(GPT2Small)
optim = AdamW(get_parameters(model), lr=3e-4)
losses = []
for i in (t := trange(100)):
    optim.zero_grad()
    logits, loss = model(x,y)
    losses.append(loss.numpy())
    loss.backward()
    optim.step()

    t.set_description(
        f"train loss: {loss.numpy():.2f}"
    )
```
```text
train loss: 0.02: 100%|██████████| 100/100 [00:39<00:00,  2.51it/s]
```

```python
plt.plot(losses)
```

![overfitting loss curve](/overfitting_loss.png)

Now for general training.

```python
class DataLoaderLite:
    def __init__(self, B, T, file_path):
        self.B=B
        self.T=T

        self.batch = lambda x: x.view(B,T)

        with open(file_path, 'r') as f:
            text = f.read()

        enc = tiktoken.get_encoding('gpt2')

        tokens = enc.encode(text)
        self.tokens = Tensor(tokens, dtype=dtypes.long)

        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T

        buf = self.tokens[self.current_position:self.current_position + B*T+1]
        x = self.batch(buf[:-1])
        y = self.batch(buf[1:])
        self.current_position += B*T

        if self.current_position + (B*T+1) > len(self.tokens):
            print("read entire document, resetting position...")
            self.current_position = 0

        return x,y
```

```python
Tensor.training = True
Tensor.no_grad = False
model = GPT2(GPT2Small)
optim = AdamW(get_parameters(model), lr=3e-4)
dl = DataLoaderLite(4, 32, "datasets/shake.txt")
losses = []
for i in (t := trange(100)):
    x, y = dl.next_batch() 
    optim.zero_grad()
    logits, loss = model(x,y)
    losses.append(loss.numpy())
    loss.backward()
    optim.step()

    t.set_description(
        f"train loss: {loss.numpy():.2f}"
    )
```

![GPT2 initial dataloaderlite loss](/dataloaderlite_loss.png)

Sweet! We now have a GPT2 model that we can run with pretrained weights and train. That's all for now.

Here's the code for training in full.

```python
from tinygrad import Tensor, dtypes
from tinygrad.nn import Embedding, Linear, LayerNorm
from tinygrad.nn.state import torch_load, load_state_dict, get_state_dict, get_parameters
from tqdm import tqdm, trange
from tinygrad.nn.optim import AdamW
from dataclasses import dataclass
from tinygrad.helpers import fetch
import tiktoken
import numpy as np
import os
import matplotlib.pyplot as plt

@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    norm_eps: float = 1e-5

@dataclass
class GPT2Small(GPT2Config):
    pass

@dataclass
class GPT2Medium(GPT2Config):
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1024

@dataclass
class GPT2Large(GPT2Config):
    n_layer: int = 36
    n_head: int = 20
    n_embd: int = 1280

@dataclass
class GPT2XL(GPT2Config):
    n_layer: int = 48
    n_head: int = 25
    n_embd: int = 1600
    
MODEL_CONFIGS = {
    'gpt2': GPT2Small,
    'gpt2-medium': GPT2Medium,
    'gpt2-large': GPT2Large,
    'gpt2-xl': GPT2XL
}

class MLP:
    def __init__(self, config : GPT2Config):
        self.c_fc = Linear(config.n_embd, config.n_embd*4)
        self.c_proj = Linear(config.n_embd*4, config.n_embd)
    def __call__(self, x):
        x = self.c_fc(x).gelu()
        x = self.c_proj(x)
        return x

class Attention:
    def __init__(self, config: GPT2Config):
        self.config = config
        self.c_attn = Linear(config.n_embd, config.n_embd*3)
        self.c_proj = Linear(config.n_embd, config.n_embd)
    def __call__(self, x):
        B,T,C = x.shape

        q, k, v = self.c_attn(x).split(C, dim=-1) #(B,T,3C) -> (B,T,C) x 3
        split_heads = lambda x: x.view(B, T, self.config.n_head, self.config.n_embd//self.config.n_head).transpose(1,2)
        q, k, v = map(split_heads, (q,k,v))

        y = q.scaled_dot_product_attention(k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y

class TransformerBlock:
    def __init__(self, config : GPT2Config):
        self.ln_1 = LayerNorm(config.n_embd, eps=config.norm_eps)
        self.ln_2 = LayerNorm(config.n_embd, eps=config.norm_eps)
        self.attn = Attention(config)
        self.mlp = MLP(config)
    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2:
    def __init__(self, config : GPT2Config = GPT2Small):
        self.config = config

        self.wte = Embedding(config.vocab_size, config.n_embd)
        self.wpe = Embedding(config.block_size, config.n_embd)
        self.h = [TransformerBlock(config) for _ in range(config.n_layer)]
        self.ln_f = LayerNorm(config.n_embd, config.norm_eps)
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)

        # tie weights - HUGE SAVINGS
        self.lm_head.weight = self.wte.weight
    
    def __call__(self, idx, targets=None):
        B,T = idx.shape

        assert T <= self.config.block_size, f"Cannot forward, model block size is {self.config.block_size} but got sequence of length {T}"
        pos = Tensor.arange(0, T, dtype=dtypes.long) # (T,)
        pos_emb = self.wpe(pos) # (T,) -> (T,C)
        tok_emb = self.wte(idx) # (B,T) -> (B,T,C)

        x = tok_emb + pos_emb
        x = x.sequential(self.h)

        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,C) -> (B,T,V)

        if targets is not None:
            loss = logits.flatten(0,1).sparse_categorical_crossentropy(targets.flatten())
            return logits, loss.realize()

        return logits, None

    @staticmethod
    def build(MODEL_NAME):
        
        weights = torch_load(fetch(f'https://huggingface.co/{MODEL_NAME}/resolve/main/pytorch_model.bin'))

        transposed = ('attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight')
        for k in weights:
            if k.endswith(transposed):
                weights[k] = weights[k].T

        weights['lm_head.weight'] = weights['wte.weight']
        model = GPT2(MODEL_CONFIGS[MODEL_NAME])
        load_state_dict(model, weights)

        return model

class DataLoaderLite:
    def __init__(self, B, T, file_path):
        self.B=B
        self.T=T

        self.batch = lambda x: x.view(B,T)

        with open(file_path, 'r') as f:
            text = f.read()

        enc = tiktoken.get_encoding('gpt2')

        tokens = enc.encode(text)
        self.tokens = Tensor(tokens, dtype=dtypes.long)

        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T

        buf = self.tokens[self.current_position:self.current_position + B*T+1]
        x = self.batch(buf[:-1])
        y = self.batch(buf[1:])
        self.current_position += B*T

        if self.current_position + (B*T+1) > len(self.tokens):
            print("read entire document, resetting position...")
            self.current_position = 0

        return x,y

Tensor.training = True
Tensor.no_grad = False
model = GPT2(GPT2Small)
optim = AdamW(get_parameters(model), lr=3e-4)
dl = DataLoaderLite(4, 32, "datasets/shake.txt")
losses = []
for i in (t := trange(100)):
    x, y = dl.next_batch() 
    optim.zero_grad()
    logits, loss = model(x,y)
    losses.append(loss.numpy())
    loss.backward()
    optim.step()

    t.set_description(
        f"train loss: {loss.numpy():.2f}"
    )
```
