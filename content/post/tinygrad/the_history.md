+++
title = 'Why Tiny?'
date = 2024-07-04T21:51:02+02:00
draft = false
+++

# TinyPhilosophy

TinyGrad is a tensor automatic differentiation library created by [George Hotz](https://en.wikipedia.org/wiki/George_Hotz) in [2020](https://www.youtube.com/watch?v=Xtws3-Pk69o&list=PLzFUMGbVxlQsh0fFZ2QKOBY25lz04A3hi). It's been described (by TinyCorp) as a middleground between Anrej Karpathy's famous micrograd project and full-blown PyTorch. It offers both the beautiful simplicity, leanness, and ease of development of micrograd, and *almost* all the speed and functionality of PyTorch.

An interesting features of TinyGrad's development is a continued, explicit constraint on the line count (~8000 LOC today). *Generally*, I consider this an ingenious design choice. Why generally? See below.

![heinous function](/tiny_oneliners.png)

Despite the sometimes unsavoury one-liners, I support the low LOC constraint because it forces you to express the logic of the underlying concepts as concisely as possible. This means no fluff, no bloat, no boilerplate. As a result, understanding the core of TinyGrad is essentially like understanding tensor automatic differentiation itself. There's minimal extra abstraction between you and the fundamental concepts. The same cannot be said for PyTorch.

I first realized that TinyGrad may be my deep learning library of choice when I tried to add support for the 1D convolution using FFT for Metal Performance Shaders in PyTorch. Such a task wouldn't just demand a solid grasp of the core principles; It would require grappling with layers of library-specific complexity. As we dive into the internals in the coming posts, you will begin to see how this is simply not an issue when developing with TinyGrad. Don't get me wrong. Things still get complicated, but they're really only complicated when *the subject itself is complicated*.

# TinyProp

I think you get the point now. TinyGrad is beautifully simple. But deep learning isn't about beautiful, simple software, it's about speed and accuracy. So what is the immediate value proposition of TinyGrad? Here are some thoughts:


1. API - similar to Torch, but way better in many areas
2. Accelerator Support - much better support for non-CPU/CUDA libraries than anything else.
3. Adding Accelerators - TinyGrad delineates frontend tensor/kernel fusion logic from backend accelerator logic with a fundamental set of 25 operations. To configure your accelerator with TinyGrad, you don't need to do too much more than define how these operations execute on it.
4. Great Community - the TinyGrad discord is active and willing to help

From [tinygrad.org](tinygrad.org)
> How is tinygrad faster than PyTorch?
> For most use cases it isn't yet, but it will be. It has three advantages:
> 1. It compiles a custom kernel for every operation, allowing extreme shape specialization.
> 2. All tensors are lazy, so it can aggressively fuse operations.
> 3. The backend is 10x+ simpler, meaning optimizing one kernel makes everything fast.

# TinyFuture

In the words of George Hotz:

>  We will beat pytorch at speed, API simplicity, and having less bugs. If we do that, we win.
