+++
title = 'Part 2 - GPT2 à la Karpathy -- but tiny'
date = 2024-06-11T16:25:35+02:00
draft = false
+++

We left off with a concise 187-line program for training a simple, unoptimized GPT-2.

Since then, I've transformed it into a more [full-fledged repo](https://github.com/nichoffs/gpt2/tree/main).

It's not necessary to go into all the little details, there are couple crucial changes.
Some are direct adaptations of Andrej Karpathy's code, while others I made myself to 
optimize the training process.

1. TinyStories Dataset - Instead of training on Tiny-Shakespeare or FineWebEdu10B, I
decided to go with TinyStories. It's synthetically generated and only includes a very
limited, simplistic vocabulary that a young child could understand. By training an LLM 
on a much more constrained set of simple stories, it can learn to generate coherent English
much more efficiently. This lets me scale down the GPT to be trainable on a single A100
for just a couple of hours.

Here's the config used for training, which is based on parameters from the paper and 
forums online:

```python
@dataclass
class TinyStories(GPT2Config):
    block_size: int = 256
    vocab_size: int = 50304
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    norm_eps: float = 1e-5
```

2. Learning-Rate Scheduler - My script uses the same exact learning rate scheduler as
Karpathy -- a linear warmup and cosine decay. 

```python
def get_lr(it, max_steps):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return Tensor([max_lr * (it + 1) / warmup_steps], requires_grad=False)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return Tensor([min_lr], requires_grad=False)
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )  # coeff starts at 1 and goes to 0
    return Tensor([min_lr + coeff * (max_lr - min_lr)], requires_grad=False)
```

3. Residual Scaling - The GPT-2 paper outlines an initialization scheme to account for 
the transformers deep residual structure:
> A modified initialization which accounts
> for the accumulation on the residual path with model depth
> is used. We scale the weights of residual layers at initialization by a factor of 1/√N where N is the number of residual layers. 

To implement this, I created a `parameters` attribute for each component of the model.
When you call get_parameters in TinyGrad, it returns a list of Tensors, without any
information about the type of layer which it's being used in. Since I need to selectively
initialize Embedding and Linear layers, I added the following:

```python
class GPT2:
    @property
    def parameters(self):
        parameters = [self.wte, self.wpe, self.ln_f, self.lm_head]
        for block in self.h:
            parameters.extend(block.parameters)
        return parameters
```

```python
class TransformerBlock:
    @property
    def parameters(self):
        return [self.ln_1, self.ln_2, *self.attn.parameters, *self.mlp.parameters]
```
```python
class Attention:
    @property
    def parameters(self):
        return [self.c_attn, self.c_proj]
```
```python
class MLP:
    @property
    def parameters(self):
        return [self.c_fc, self.c_proj]
```

And this for initialization...

```python
    def init_weights(self, param):
        if isinstance(param, nn.Linear):
            std = 0.02
            if hasattr(param, "RESIDUAL_SCALING"):
                std *= (2 * self.config.n_layer) ** -0.5
            param.weight = Tensor.normal(
                param.weight.shape,
                mean=0,
                std=std,
            )
            if param.bias is not None:
                param.bias = Tensor.zeros_like(param.bias)
        elif isinstance(param, nn.Embedding):
            param.weight = Tensor.normal(param.weight.shape, mean=0, std=0.02)

```

4. Selective Weight Decay

    Notice how I use get_parameters for this one because we just want to get Tensors.

```python
    def configure_optimizers(self, lr, b1, b2, eps, wd):

        parameters = get_parameters(self)

        params_nodecay = [
            param
            for param in parameters
            if len(param.shape) < 2
            and (param.requires_grad or param.requires_grad is None)
        ]
        params_decay = [
            param
            for param in parameters
            if len(param.shape) >= 2
            and (param.requires_grad or param.requires_grad is None)
        ]

        opt_decay = AdamW(params_decay, lr=lr, b1=b1, b2=b2, eps=eps, weight_decay=wd)
        opt_nodecay = AdamW(
            params_nodecay, lr=lr, b1=b1, b2=b2, eps=eps, weight_decay=0
        )

        num_params_decay = sum(param.numel() for param in opt_decay.params)
        num_params_nodecay = sum(param.numel() for param in opt_nodecay.params)

        print(
            f"num decay params: {num_params_decay} num nodecay params: {num_params_nodecay}"
        )

        optim_group = OptimizerGroup(opt_decay, opt_nodecay)
        return optim_group
```

5. Checkpointing - The model will save its state every couple hundred epochs and can 
be loaded.

6. Validation Loss

I trained this model with a batch size of 64 for the first 10,000 epochs and 128 for the final ~5000.

This is a sample generation with prompt "Once upon a time" compared to GPT2-medium:

```text
Pretrained Generation: Once upon a time it was all about a team of robots or robots and I would have gotten a better joke out of them than this, and when it came to the actual robot, I had no idea what the whole plan was about.

Checkpointed Generation: Once upon a time, there was a girl called Kitty. Grace had an exciting time. One day, Blue was playing in the park. She had a pretty coat and a picture full of a big blue ball. It was an unusual day, and
```
